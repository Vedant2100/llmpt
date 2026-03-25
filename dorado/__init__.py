"""Dorado: Two-stage reasoning post-training with dual-reward refinement."""

from dorado.utils import (
    clear_gpu,
    set_random_seeds,
    cleanup_storage,
    enforce_storage_budget,
    pipeline_warn,
    drain_pipeline_warnings,
)
from dorado.config import (
    PROFILES,
    get_profile,
    build_experiment_grid,
    make_bnb_config,
    make_model_load_kwargs,
    make_results_paths,
)


# ── Lazy imports for modules with heavy deps (torch, trl, peft) ──────
# This prevents import-time crashes when torch/trl versions are mismatched.
# The actual imports happen only when the functions are called.

def run_sft_stage(*args, **kwargs):
    from dorado.sft import run_sft_stage as _fn
    return _fn(*args, **kwargs)

def run_candidate_generation(*args, **kwargs):
    from dorado.generation import run_candidate_generation as _fn
    return _fn(*args, **kwargs)

def run_labeling_stage(*args, **kwargs):
    from dorado.labeling import run_labeling_stage as _fn
    return _fn(*args, **kwargs)

def run_dpo_training(*args, **kwargs):
    from dorado.dpo import run_dpo_training as _fn
    return _fn(*args, **kwargs)

def run_full_evaluation(*args, **kwargs):
    from dorado.evaluation import run_full_evaluation as _fn
    return _fn(*args, **kwargs)

def evaluate_model(*args, **kwargs):
    from dorado.evaluation import evaluate_model as _fn
    return _fn(*args, **kwargs)

def run_single_experiment(*args, **kwargs):
    from dorado.orchestrator import run_single_experiment as _fn
    return _fn(*args, **kwargs)

def run_all_experiments(*args, **kwargs):
    from dorado.orchestrator import run_all_experiments as _fn
    return _fn(*args, **kwargs)

def cleanup_artifacts(*args, **kwargs):
    from dorado.orchestrator import cleanup_artifacts as _fn
    return _fn(*args, **kwargs)
