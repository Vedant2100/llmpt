"""Shared utilities: GPU cleanup, seeding, answer extraction, storage."""

import gc
import os
import re
import glob
import shutil
import random
import subprocess
from pathlib import Path

import torch
import numpy as np
import importlib
import sys


def _get_protected_artifact_paths() -> set[str]:
    """Read protected artifact paths from env and normalize to absolute paths."""
    raw = os.environ.get("DORADO_PROTECTED_ARTIFACTS", "").strip()
    if not raw:
        return set()
    parts = [p.strip() for p in raw.split(os.pathsep) if p.strip()]
    return {os.path.abspath(p) for p in parts}


def _safe_remove_path(path: str, protected_abs_paths: set[str]) -> bool:
    """Remove path unless protected. Returns True when removed."""
    abs_path = os.path.abspath(path)
    if abs_path in protected_abs_paths:
        return False
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        return True
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def get_mixed_precision_kwargs() -> dict:
    """Return safe mixed-precision flags for HF Trainer configs."""
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    if torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": True}


def clear_gpu():
    """Clear GPU memory, resilient to corrupted CUDA context."""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"⚠️ CUDA cleanup warning (non-fatal): {e}")


def set_random_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Random seeds set to {seed}")


def extract_answer_from_response(text: str) -> str:
    """
    Fallback answer extraction (GSM8K-style #### markers).

    Prefer ``eval/utils/parser.py:extract_answer`` for MATH-style ``\\boxed{}``
    extraction.  This function is kept only as a safety net when eval utils
    are unavailable.
    """
    if "####" in text:
        text = text.split("####")[-1].strip()
    text = text.replace(",", "")
    nums = re.findall(r"-?\d+(?:/\d+|\.\d+)?", text)
    if not nums:
        return "None"
    return _canonicalize_numeric_token(nums[-1])




def _canonicalize_numeric_token(token: str) -> str:
    """Normalize integer/decimal/fraction tokens to a stable comparable string."""
    token = token.strip()
    if "/" in token:
        try:
            from fractions import Fraction

            value = float(Fraction(token))
            if abs(value - round(value)) < 1e-9:
                return str(int(round(value)))
            out = f"{value:.12f}".rstrip("0").rstrip(".")
            return out if out else "0"
        except Exception:
            return token
    try:
        value = float(token)
    except ValueError:
        return token
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    out = f"{value:.12f}".rstrip("0").rstrip(".")
    return out if out else "0"


# ── pipeline warning accumulator ─────────────────────────────────────

_pipeline_warnings: list[str] = []


def pipeline_warn(msg: str):
    """Record a pipeline warning and print it immediately."""
    _pipeline_warnings.append(msg)
    print(f"⚠️  [WARNING] {msg}")


def drain_pipeline_warnings() -> list[str]:
    """Return all accumulated warnings and clear the buffer."""
    w = _pipeline_warnings.copy()
    _pipeline_warnings.clear()
    return w


def cleanup_storage():
    """Aggressive cleanup to free disk space (pip cache, HF cache, artifacts)."""
    print("🧹 Purging caches & training artifacts...")
    protected_abs_paths = _get_protected_artifact_paths()

    subprocess.run(
        ["pip", "cache", "purge"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    hf_roots = {
        os.path.expanduser("~/.cache/huggingface"),
        os.environ.get("HF_HOME", ""),
        os.environ.get("HF_HUB_CACHE", ""),
        os.environ.get("HF_DATASETS_CACHE", ""),
        os.environ.get("TRANSFORMERS_CACHE", ""),
    }
    for root in [p for p in hf_roots if p]:
        if os.path.exists(root):
            shutil.rmtree(root, ignore_errors=True)
            print(f"  ✓ Cleared cache: {root}")

    runtime_cache = os.environ.get("DORADO_RUNTIME_CACHE", "")
    if runtime_cache and os.path.exists(runtime_cache):
        shutil.rmtree(runtime_cache, ignore_errors=True)
        print(f"  ✓ Cleared runtime cache: {runtime_cache}")

    artifacts = [
        "coldstart_dorado*",
        "dorado_final*",
        "dorado_round_*",
        "runs/",
    ]
    skipped = 0
    for pattern in artifacts:
        for path in glob.glob(pattern):
            removed = _safe_remove_path(path, protected_abs_paths)
            if not removed:
                skipped += 1
    if skipped:
        print(f"  ✓ Preserved {skipped} protected active artifact(s)")
    print("  ✓ Removed training artifacts")

    total, used, free = shutil.disk_usage("/")
    print(f"\n📊 Disk: {used/total:.1%} used  ({free/1024**3:.2f} GB free)")
    if free / 1024**3 < 2:
        print("   ⚠️ Still low on space")
    else:
        print("   ✅ Sufficient space")


def _path_size_gb(path: str) -> float:
    """Return directory/file size in GiB (best effort)."""
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / 1024**3
    total = 0
    for child in p.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total / 1024**3


def enforce_storage_budget(
    max_home_gb: float = 9.5,
    min_free_gb: float = 2.0,
    hard_fail: bool = True,
):
    """Keep storage usage below limits by pruning caches/artifacts proactively.

    - ``max_home_gb``: soft cap for ``/home/jovyan`` usage.
    - ``min_free_gb``: minimum free GiB required on filesystem.
    """
    home_dir = "/home/jovyan"
    home_used = _path_size_gb(home_dir)
    total, used, free = shutil.disk_usage("/")
    free_gb = free / 1024**3

    if home_used <= max_home_gb and free_gb >= min_free_gb:
        return

    print(
        "🧯 Storage guard triggered: "
        f"home_used={home_used:.2f}GiB (cap {max_home_gb:.2f}), "
        f"free={free_gb:.2f}GiB (min {min_free_gb:.2f})"
    )
    protected_abs_paths = _get_protected_artifact_paths()

    # Step 1: remove run artifacts first
    for pattern in (
        "coldstart_dorado*",
        "dorado_final*",
        "dorado_round_*",
    ):
        for path in glob.glob(pattern):
            _safe_remove_path(path, protected_abs_paths)

    # Step 2: clear caches (home + redirected)
    cleanup_storage()

    # Re-check
    home_used = _path_size_gb(home_dir)
    _total2, _used2, free2 = shutil.disk_usage("/")
    free_gb = free2 / 1024**3
    ok = home_used <= max_home_gb and free_gb >= min_free_gb
    if ok:
        print(
            f"✅ Storage guard satisfied: home_used={home_used:.2f}GiB, free={free_gb:.2f}GiB"
        )
        return

    msg = (
        "Storage budget still exceeded after cleanup: "
        f"home_used={home_used:.2f}GiB (cap {max_home_gb:.2f}), "
        f"free={free_gb:.2f}GiB (min {min_free_gb:.2f})."
    )
def is_deepspeed_functional() -> bool:
    """Check if DeepSpeed is installed and has a working CUDA compiler/libaio.

    Returns False if DeepSpeed is missing, or if it would crash on metadata checks
    due to missing hardware compilers (common on JupyterHub).
    """
    if shutil.which("nvcc") is None and os.environ.get("CUDA_HOME") == "/usr/local/cuda":
        # Known broken environment: has CUDA folder but no compiler.
        # DeepSpeed op_builder will crash on import if we don't handle this.
        return False
    
    try:
        importlib.import_module("deepspeed")
        return True
    except Exception:
        return False


def harden_environment():
    """Surgical environment fix to prevent DeepSpeed import crashes on crippled hardware.

    - Unsets CUDA_HOME if nvcc is missing.
    - Monkeypatches DeepSpeed op_builder to prevent FileNotFoundError/MissingCUDAException.
    - Checks for libaio and warns if missing (breaks DeepSpeed async-io).
    """
    # 1. Fix CUDA_HOME ghosting
    if shutil.which("nvcc") is None:
        if "CUDA_HOME" in os.environ:
            # Only unset if it points to the standard location which we know is broken on this hub
            if os.environ["CUDA_HOME"] == "/usr/local/cuda":
                del os.environ["CUDA_HOME"]
        
        # 2. Monkeypatch DeepSpeed OpBuilder if already installed to prevent crash-on-import
        # We do this by swapping out the builder check with a dummy that says "not compatible"
        try:
            # We must do this BEFORE any other dorado modules import transformers/trl
            import deepspeed.ops.op_builder.builder as ds_builder
            
            orig_check = ds_builder.OpBuilder.is_compatible
            def safe_is_compatible(self, verbose=False):
                try:
                    return orig_check(self, verbose)
                except Exception:
                    # If it would have crashed (MissingCUDAException), just say No
                    return False
            
            ds_builder.OpBuilder.is_compatible = safe_is_compatible
            
            # Also patch the global version check
            def safe_cuda_version():
                try:
                    return ds_builder.installed_cuda_version()
                except Exception:
                    return (0, 0)
            ds_builder.installed_cuda_version = safe_cuda_version
            
        except (ImportError, Exception):
            pass

    # 3. Check libaio (required for DS async-io, often missing on Hubs)
    if shutil.which("ldconfig"):
        try:
            res = subprocess.check_output(["ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL)
            if "libaio.so" not in res:
                os.environ["DS_SKIP_CUDA_CHECK"] = "1" # Help DS be less picky
        except Exception:
            pass

    # 4. Torch optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
