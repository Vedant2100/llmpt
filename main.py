#!/usr/bin/env python3
"""Dorado: Two-Stage Reasoning Post-Training — Main Entry Point.

Usage:
    python main.py                          # default: smoke profile
    python main.py --profile fast           # quick experiments on 1.5B
    python main.py --profile full           # paper-faithful 7B run
    python main.py --profile fast --override math_prompt_count=1000
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ── Environment setup (must happen before torch import) ──────────────

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Writable runtime cache (JupyterHub-safe)
repo_candidates = ["/home/jovyan/llmpt", os.getcwd()]
repo_root = next((p for p in repo_candidates if os.path.isdir(p)), os.getcwd())

cache_candidates = [
    os.path.join(repo_root, ".runtime_cache"),
    "/tmp/llmpt_runtime_cache",
]

runtime_cache = None
for cand in cache_candidates:
    try:
        Path(cand).mkdir(parents=True, exist_ok=True)
        test_file = Path(cand) / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        runtime_cache = cand
        break
    except Exception:
        continue

if runtime_cache is None:
    raise RuntimeError("No writable runtime cache path found.")

os.environ["DORADO_RUNTIME_CACHE"] = runtime_cache
os.environ["HF_HOME"] = f"{runtime_cache}/huggingface"
os.environ["HF_HUB_CACHE"] = f"{runtime_cache}/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = f"{runtime_cache}/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{runtime_cache}/huggingface/transformers"
os.environ["PIP_CACHE_DIR"] = f"{runtime_cache}/pip"

# Clear stale caches
for stale in ["/home/jovyan/.cache/huggingface", "/home/jovyan/.cache/pip"]:
    if os.path.exists(stale):
        shutil.rmtree(stale, ignore_errors=True)


def pick_free_gpus(min_free_gb: int = 16, max_gpus: int = 8) -> list[str]:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ]
        rows = subprocess.check_output(cmd, text=True).strip().splitlines()
        parsed = []
        for row in rows:
            idx_str, free_mb_str = [x.strip() for x in row.split(",")[:2]]
            parsed.append((idx_str, int(free_mb_str)))

        keep = [x for x in parsed if x[1] >= min_free_gb * 1024]
        if not keep:
            keep = sorted(parsed, key=lambda x: x[1], reverse=True)[:max_gpus]
        else:
            keep = sorted(keep, key=lambda x: x[1], reverse=True)[:max_gpus]

        return [idx for idx, _ in keep]
    except Exception:
        return ["0"]


selected = pick_free_gpus(min_free_gb=16, max_gpus=8)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected)

if len(selected) > 1:
    os.environ["DORADO_MAX_MEMORY_PER_GPU"] = "22GiB"


# ── Ensure dorado is importable ──────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


# ── Main ─────────────────────────────────────────────────────────────

def parse_overrides(override_strs: list[str] | None) -> dict:
    """Parse 'key=value' override strings into a dict."""
    if not override_strs:
        return {}
    overrides = {}
    for s in override_strs:
        if "=" not in s:
            print(f"⚠️ Ignoring malformed override: {s}")
            continue
        key, val = s.split("=", 1)
        # Try to interpret value as int/float/bool
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        overrides[key.strip()] = val
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Dorado: Two-Stage Reasoning Post-Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "fast", "full"],
        default="smoke",
        help="Experiment profile (default: smoke)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        metavar="KEY=VALUE",
        help="Override profile settings, e.g. --override math_prompt_count=1000 sft_samples=500",
    )
    args = parser.parse_args()

    # ── Print environment info ───────────────────────────────────────
    print(f"🧩 CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")
    print(f"📁 Repo root: {repo_root}")
    print(f"💾 Runtime cache: {runtime_cache}")

    import torch

    if torch.cuda.is_available():
        print(f"🎯 Primary GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔢 GPU count: {torch.cuda.device_count()}")
    else:
        print("⚠️ No CUDA GPUs available")

    # ── Import dorado ────────────────────────────────────────────────
    from dorado import (
        build_experiment_grid,
        make_results_paths,
        run_all_experiments,
        cleanup_artifacts,
        PROFILES,
    )

    print(f"✅ dorado loaded (profiles: {list(PROFILES.keys())})")

    # ── Build experiments ────────────────────────────────────────────
    overrides = parse_overrides(args.override)
    experiments = build_experiment_grid(profile=args.profile, overrides=overrides or None)
    _, results_file, checkpoint_file = make_results_paths()

    cfg = experiments[0]
    print(f"\n{'─'*60}")
    print(f"Profile:       {args.profile}")
    print(f"Base model:    {cfg['base_model']}")
    print(f"Finetuning:    {cfg['finetuning_type']}")
    print(f"SFT data:      {cfg['sft_dataset_name']} ({cfg['sft_samples']} samples)")
    print(f"Math prompts:  {cfg['math_prompt_count']}")
    print(f"RM strategy:   {cfg['rm_strategy']}")
    print(f"DPO β/lr:      {cfg['dpo_beta']}/{cfg['dpo_lr']}")
    print(f"Eval engine:   {cfg['eval_engine']}")
    print(f"Eval benchmarks: {cfg['eval_benchmarks']}")
    print(f"Experiments:   {len(experiments)}")
    print(f"Results:       {results_file}")
    if overrides:
        print(f"Overrides:     {overrides}")
    print(f"{'─'*60}\n")

    # ── Run ──────────────────────────────────────────────────────────
    results_df = run_all_experiments(experiments, results_file, checkpoint_file)

    # ── Summary ──────────────────────────────────────────────────────
    if results_df is not None and not results_df.empty:
        display_cols = [
            c
            for c in results_df.columns
            if "accuracy" in c
            or c in ["experiment_id", "status", "runtime_minutes", "improvement_over_base"]
        ]
        print("\n" + results_df[display_cols].to_string(index=False))

    # ── Cleanup ──────────────────────────────────────────────────────
    cleanup_artifacts()
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
