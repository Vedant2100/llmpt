"""Dorado: Iterative DPO training pipeline for math reasoning."""

from dorado.utils import (
    clear_gpu,
    set_random_seeds,
    extract_answer_from_response,
    cleanup_storage,
)
from dorado.config import (
    DATASET_CONFIG,
    MODEL_CONFIG,
    ARCHITECTURE_CONFIG,
    TRAINING_CONFIG,
    GENERATION_CONFIG,
    EVAL_CONFIG,
    DUAL_PREFERENCE_CONFIG,
    build_experiment_grid,
    estimate_time,
    make_results_paths,
)
from dorado.sft import run_sft_stage
from dorado.generation import run_candidate_generation
from dorado.labeling import run_labeling_stage
from dorado.reward_model import run_rm_training
from dorado.dpo import run_dpo_training
from dorado.evaluation import run_full_evaluation
from dorado.orchestrator import (
    run_single_experiment,
    run_all_experiments,
    cleanup_artifacts,
)
