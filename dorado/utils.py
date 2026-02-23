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
    Unified answer extraction for GSM8K.

    Returns the last number found in *text* (after ``####`` if present),
    or ``"None"`` when no number is found.  Must be used consistently
    during both training and evaluation.
    """
    if "####" in text:
        text = text.split("####")[-1].strip()
    text = text.replace(",", "")  # normalise "1,200" → "1200"
    nums = re.findall(r"\d+", text)
    return nums[-1] if nums else "None"


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
        "reward_model*",
        "coldstart_dorado*",
        "dorado_final*",
        "dorado_round_*",
        "runs/",
    ]
    for pattern in artifacts:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
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

    # Step 1: remove run artifacts first
    for pattern in (
        "reward_model*",
        "coldstart_dorado*",
        "dorado_final*",
        "dorado_round_*",
    ):
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)

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
    if hard_fail:
        raise RuntimeError(msg)
    print(f"⚠️ {msg}")
