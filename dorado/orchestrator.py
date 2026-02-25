"""Experiment orchestrator: single-experiment runner and full sweep."""

import gc
import glob
import os
import shutil
import time
import traceback

import pandas as pd
import torch

from dorado.utils import (
    clear_gpu,
    set_random_seeds,
    drain_pipeline_warnings,
    enforce_storage_budget,
)
from dorado.sft import run_sft_stage
from dorado.generation import run_candidate_generation
from dorado.labeling import run_labeling_stage
from dorado.reward_model import run_rm_training
from dorado.dpo import run_dpo_training
from dorado.evaluation import run_full_evaluation
from dorado.config import (
    DATASET_CONFIG,
    MODEL_CONFIG,
    ARCHITECTURE_CONFIG,
    TRAINING_CONFIG,
    GENERATION_CONFIG,
    EVAL_CONFIG,
    DUAL_PREFERENCE_CONFIG,
)


# ── artifact cleanup ─────────────────────────────────────────────────


def cleanup_artifacts():
    """Remove intermediate training artifacts while preserving results outputs."""
    artifact_patterns = (
        "coldstart_*",
        "reward_model*",
        "dorado_final*",
        "dorado_round_*",
    )
    removed_paths: list[str] = []
    for pattern in artifact_patterns:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                removed_paths.append(path)
            elif os.path.exists(path):
                os.remove(path)
                removed_paths.append(path)
    if removed_paths:
        print(
            "🧹 Cleanup removed "
            f"{len(removed_paths)} intermediate artifact(s): "
            + ", ".join(sorted(removed_paths))
        )
    else:
        print("🧹 Cleanup found no intermediate artifacts to remove")
    try:
        clear_gpu()
    except Exception as e:
        print(f"⚠️ GPU cleanup warning (non-fatal): {e}")


# ── single experiment ────────────────────────────────────────────────


def run_single_experiment(exp_config: dict) -> dict:
    """Execute one full experiment (SFT → iterative DPO → eval)."""
    exp_id = exp_config["experiment_id"]
    print(f"\n{'='*70}\nEXPERIMENT {exp_id}\n{'='*70}")

    print("🧹 Clearing intermediate artifacts before experiment run…")
    cleanup_artifacts()

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 1024**3
            res = torch.cuda.memory_reserved() / 1024**3
            print(f"\n🎮 GPU Memory: {alloc:.2f} GB allocated, {res:.2f} GB reserved")
            if alloc > 1.0:
                print("⚠️  >1 GB already allocated – running aggressive cleanup…")
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"⚠️ CUDA status check warning: {e}")

    set_random_seeds(exp_config.get("random_seed", 42))
    enforce_storage_budget()

    results: dict = {"experiment_id": exp_id, "status": "in_progress", "error": None}

    def set_protected_artifacts(paths: list[str]):
        """Protect active adapter artifacts from storage-guard deletion."""
        normalized = [p for p in paths if p]
        if normalized:
            os.environ["DORADO_PROTECTED_ARTIFACTS"] = os.pathsep.join(normalized)
        elif "DORADO_PROTECTED_ARTIFACTS" in os.environ:
            del os.environ["DORADO_PROTECTED_ARTIFACTS"]

    # Drain any stale warnings from a previous run
    drain_pipeline_warnings()
    start = time.time()

    try:
        # Stage 1: SFT
        enforce_storage_budget()
        print("\n[Stage 1/6] Cold-Start SFT…")
        sft_path = run_sft_stage(exp_config)
        set_protected_artifacts([sft_path])

        # Iterative DPO
        round_metrics: list[dict] = []
        prev_path = sft_path

        for rnd in range(exp_config["iterative_dpo_rounds"]):
            r = rnd + 1
            print(
                f"\n{'='*70}\nDPO ROUND {r}/{exp_config['iterative_dpo_rounds']}\n{'='*70}"
            )

            enforce_storage_budget()
            print(f"\n[Stage 2/6] Candidate Generation (Round {r})…")
            samples, gt, _qs = run_candidate_generation(exp_config, prev_path)

            enforce_storage_budget()
            print(f"\n[Stage 3/6] Initial Labeling (Round {r})…")
            init_pairs, init_labels, init_stats = run_labeling_stage(
                exp_config, samples, gt, use_rm=False
            )
            if not init_pairs:
                print(f"❌ No pairs in round {r}. Stopping.")
                break

            if exp_config["use_rm_scoring"]:
                enforce_storage_budget()
                print(f"\n[Stage 4/6] Reward Model Training (Round {r})…")
                run_rm_training(exp_config, init_pairs, init_labels)
                print(f"\n[Stage 3b/6] Re-labeling with RM (Round {r})…")
                pairs, labels, pair_stats = run_labeling_stage(
                    exp_config, samples, gt, use_rm=True
                )
            else:
                print(
                    "\n[Stage 4/6] Skipped Reward Model Training (use_rm_scoring=False)"
                )
                print("   Using verifiable correctness pairs only")
                pairs, labels, pair_stats = init_pairs, init_labels, init_stats

            enforce_storage_budget()
            dpo_out = "dorado_final" if rnd == 0 else f"dorado_round_{r}"
            print(f"\n[Stage 5/6] DPO Training (Round {r})…")
            dpo_result = run_dpo_training(exp_config, pairs, prev_path, dpo_out)
            if dpo_result is None:
                print(f"❌ DPO round {r} produced no model. Stopping iterative loop.")
                break
            prev_path = dpo_out
            set_protected_artifacts([sft_path, prev_path])

            round_metrics.append(
                {
                    "round": r,
                    "num_pairs": pair_stats["num_pairs"],
                    "correct_incorrect_pairs": pair_stats["correct_incorrect_pairs"],
                    "correct_correct_pairs": pair_stats["correct_correct_pairs"],
                    "all_wrong_dropped_questions": pair_stats.get(
                        "all_wrong_dropped_questions", 0
                    ),
                    "all_correct_questions": pair_stats.get("all_correct_questions", 0),
                    "mixed_questions": pair_stats.get("mixed_questions", 0),
                    "avg_rm_score": pair_stats["avg_rm_score"],
                }
            )

        # Stage 6: Evaluation
        enforce_storage_budget()
        print("\n[Stage 6/6] Final Evaluation…")
        model_paths = {
            "BASE": exp_config["base_model"],
            "SFT": sft_path,
            "DORADO": "dorado_final",
        }
        if exp_config["iterative_dpo_rounds"] > 1:
            for ri in range(1, exp_config["iterative_dpo_rounds"]):
                rp = f"dorado_round_{ri}"
                if os.path.exists(rp):
                    model_paths[f"DORADO_R{ri}"] = rp

        all_metrics, all_eval_results = run_full_evaluation(exp_config, model_paths)

        results["eval_examples"] = all_eval_results
        results["status"] = "success"
        results["runtime_minutes"] = (time.time() - start) / 60
        results["round_metrics"] = round_metrics

        for tag in ("BASE", "SFT", "DORADO"):
            if tag in all_metrics:
                m = all_metrics[tag]
                key = tag.lower()
                results[f"{key}_accuracy"] = m["accuracy"]
                results[f"{key}_ci_lower"] = m["ci_lower"]
                results[f"{key}_ci_upper"] = m["ci_upper"]

        if "DORADO" in all_metrics:
            results["improvement_over_base"] = results["dorado_accuracy"] - results.get(
                "base_accuracy", 0
            )
            results["improvement_over_sft"] = results["dorado_accuracy"] - results.get(
                "sft_accuracy", 0
            )

        if round_metrics:
            last = round_metrics[-1]
            results["final_num_pairs"] = last["num_pairs"]
            results["final_avg_rm_score"] = last["avg_rm_score"]

        # ── collect pipeline warnings ────────────────────────────────
        warnings = drain_pipeline_warnings()
        warnings = list(dict.fromkeys(warnings))
        results["pipeline_warnings"] = warnings
        results["warning_count"] = len(warnings)
        if warnings:
            print(f"\n{'─'*70}")
            print(f"⚠️  PIPELINE WARNINGS ({len(warnings)}):")
            for i, w in enumerate(warnings, 1):
                print(f"  {i}. {w}")
            print(f"{'─'*70}")

        print(
            f"\n✅ Experiment {exp_id} completed in {results['runtime_minutes']:.1f} min"
        )

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        results["runtime_minutes"] = (time.time() - start) / 60
        # Still collect any warnings that fired before the crash
        warnings = drain_pipeline_warnings()
        warnings = list(dict.fromkeys(warnings))
        results["pipeline_warnings"] = warnings
        results["warning_count"] = len(warnings)
        print(f"\n❌ Experiment {exp_id} failed: {e}")
        traceback.print_exc()
    finally:
        set_protected_artifacts([])

    return results


# ── run full sweep ───────────────────────────────────────────────────


def run_all_experiments(
    experiments: list[dict],
    results_file: str,
    checkpoint_file: str,
) -> pd.DataFrame:
    """Run every experiment in *experiments*, checkpoint after each one.

    Returns the final results DataFrame.
    """
    completed_ids: set[int] = set()
    results_log: list[dict] = []
    all_eval_examples: list[dict] = []

    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint: {checkpoint_file}")
        cp = pd.read_excel(checkpoint_file)
        if "status" in cp.columns:
            completed_ids = set(
                cp.loc[cp["status"] == "success", "experiment_id"].tolist()
            )
        else:
            completed_ids = set(cp["experiment_id"].tolist())
        results_log = cp.to_dict("records")
        print(f"Resuming – {len(completed_ids)} experiments already done.")
    else:
        print("No checkpoint found. Starting fresh.")

    print(
        "\n🧹 Pre-run cleanup: removing intermediate artifacts (results folder preserved)…"
    )
    cleanup_artifacts()

    print(f"\n{'='*70}\nRUNNING {len(experiments)} EXPERIMENTS\n{'='*70}\n")

    for idx, cfg in enumerate(experiments):
        eid = cfg["experiment_id"]
        if eid in completed_ids:
            print(f"Skipping experiment {eid} (already completed)")
            continue

        print(f"\n{'='*70}")
        print(f"EXPERIMENT {idx+1}/{len(experiments)} (ID: {eid})")
        print(f"{'='*70}")
        for k, v in cfg.items():
            if k != "experiment_id":
                print(f"  {k}: {v}")

        res = run_single_experiment(cfg)
        eval_examples = res.pop("eval_examples", [])
        for ex in eval_examples:
            ex["experiment_id"] = cfg["experiment_id"]
        all_eval_examples.extend(eval_examples)
        # Serialise warnings list for Excel (join as newline-separated string)
        warnings_list = res.pop("pipeline_warnings", [])
        res["pipeline_warnings"] = "\n".join(warnings_list) if warnings_list else ""
        results_log.append({**cfg, **res})

        print(f"\nCleaning up artifacts for experiment {eid}…")
        cleanup_artifacts()

        pd.DataFrame(results_log).to_excel(
            checkpoint_file, index=False, engine="openpyxl"
        )
        print(f"💾 Checkpoint saved: {checkpoint_file}")

    # ── export ───────────────────────────────────────────────────────
    if not results_log:
        print("No results to export.")
        return pd.DataFrame()

    df = pd.DataFrame(results_log)

    all_param_keys = set()
    for d in (
        DATASET_CONFIG,
        MODEL_CONFIG,
        ARCHITECTURE_CONFIG,
        TRAINING_CONFIG,
        GENERATION_CONFIG,
        EVAL_CONFIG,
        DUAL_PREFERENCE_CONFIG,
    ):
        all_param_keys |= d.keys()

    meta = ["experiment_id", "status", "runtime_minutes", "error"]
    params = [c for c in df.columns if c in all_param_keys]
    metrics = [
        c
        for c in df.columns
        if any(k in c for k in ("accuracy", "improvement", "num_pairs", "rm_score"))
    ]
    rest = [c for c in df.columns if c not in meta + params + metrics]
    cols = [c for c in meta + params + metrics + rest if c in df.columns]
    df = df[cols]

    try:
        with pd.ExcelWriter(results_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Results", index=False)
            if all_eval_examples:
                ex_df = pd.DataFrame(all_eval_examples)
                ex_df["Prompt"] = ex_df["Prompt"].str[:200]
                ex_df["Full Response"] = ex_df["Full Response"].str[:500]
                ex_cols = [
                    "experiment_id",
                    "Model",
                    "Prompt",
                    "Correct Answer",
                    "Model Answer",
                    "Accurate",
                    "Full Response",
                ]
                ex_df = ex_df[[c for c in ex_cols if c in ex_df.columns]]
                ex_df.to_excel(writer, sheet_name="Examples", index=False)
        print(f"\n{'='*70}\nRESULTS EXPORTED → {results_file}\n{'='*70}")
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("✅ Checkpoint removed (export successful)")
    except Exception as e:
        print(f"\n❌ Export error: {e}")
        print(f"⚠️ Checkpoint preserved: {checkpoint_file}")
        raise

    n_ok = sum(1 for r in results_log if r.get("status") == "success")
    n_fail = sum(1 for r in results_log if r.get("status") == "failed")
    print(f"Total: {len(results_log)} | Success: {n_ok} | Failed: {n_fail}")

    if "dorado_accuracy" in df.columns:
        print(f"\nMean DORADO accuracy: {df['dorado_accuracy'].mean():.1%}")
        print(f"Best DORADO accuracy: {df['dorado_accuracy'].max():.1%}")

    print(f"\n{'='*70}\nALL EXPERIMENTS COMPLETE!\n{'='*70}")
    return df
