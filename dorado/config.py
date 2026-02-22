"""Experiment configuration and grid builder."""

import os
import itertools
from datetime import datetime

# ── Memory optimisation ──────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 0 is full; use GPU 1 instead

# ── Parameter dictionaries (edit these lists to sweep) ───────────────
DATASET_CONFIG = {
    "sft_samples": [20],
    "dpo_pairs": [3],
    "candidates_per_question": [4],
    "sft_dataset_name": ["tatsu-lab/alpaca"],
    "eval_split": ["test"],
    "eval_max_samples": [20],
    "random_seed": [42],
}

MODEL_CONFIG = {
    "base_model": ["Qwen/Qwen2.5-0.5B"],
    "rm_base_model": ["Qwen/Qwen2.5-0.5B"],
}

ARCHITECTURE_CONFIG = {
    "lora_r": [4],
    "lora_alpha": [8],
    "dpo_beta": [0.1],
    "gradient_accumulation_steps": [4],
    "quantization_bits": [0],  # 0 = no quantization (fp16), 4 = NF4, 8 = int8
}


def make_bnb_config(exp_config: dict):
    """Build a BitsAndBytesConfig, or None if quantization is disabled."""
    import torch
    from transformers import BitsAndBytesConfig

    bits = exp_config.get("quantization_bits", 0)
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        return None  # no quantization — load in fp16/bf16 directly


TRAINING_CONFIG = {
    "iterative_dpo_rounds": [1],
    "sft_epochs": [1],
    "rm_epochs": [1],
    "dpo_epochs": [1],
    "sft_batch_size": [1],
    "rm_batch_size": [1],
    "dpo_batch_size": [1],
}

GENERATION_CONFIG = {
    "temperature": [0.7],
    "max_new_tokens_gen": [64],
    "max_new_tokens_eval": [128],
}

EVAL_CONFIG = {
    "eval_batch_size": [1],
}

DUAL_PREFERENCE_CONFIG = {
    "use_rm_scoring": [False, True],
    "rm_weight": [0.5],
    "correctness_weight": [1.0],
}


# ── Grid helpers ─────────────────────────────────────────────────────
def build_experiment_grid() -> list[dict]:
    """Generate all parameter combinations."""
    all_configs = {
        **DATASET_CONFIG,
        **MODEL_CONFIG,
        **ARCHITECTURE_CONFIG,
        **TRAINING_CONFIG,
        **GENERATION_CONFIG,
        **EVAL_CONFIG,
        **DUAL_PREFERENCE_CONFIG,
    }
    keys = list(all_configs.keys())
    values = [all_configs[k] for k in keys]

    experiments = []
    for i, combo in enumerate(itertools.product(*values)):
        exp = dict(zip(keys, combo))
        exp["experiment_id"] = i
        experiments.append(exp)
    return experiments


def estimate_time(exp: dict) -> dict:
    """Rough runtime estimate (minutes)."""
    sft_min = 0.004 * exp["sft_samples"] * exp["sft_epochs"]
    gen_min = 0.002 * exp["dpo_pairs"] * (exp["candidates_per_question"] / 2)
    rm_min = 0.003 * exp["dpo_pairs"] * exp["rm_epochs"]
    dpo_min = 0.003 * exp["dpo_pairs"] * exp["dpo_epochs"]
    eval_min = 0.01 * exp.get("eval_max_samples", 100) / exp["eval_batch_size"]
    rounds = max(1, exp["iterative_dpo_rounds"])
    total = sft_min + (gen_min + rm_min + dpo_min) * rounds + eval_min
    return dict(
        sft_min=sft_min,
        gen_min=gen_min,
        rm_min=rm_min,
        dpo_min=dpo_min,
        eval_min=eval_min,
        total_min=total,
    )


def make_results_paths(base_dir: str = "results") -> tuple[str, str, str]:
    """Return (results_dir, results_file, checkpoint_file)."""
    results_date = datetime.now().strftime("%Y-%m-%d")
    results_dir = f"{base_dir}/{results_date}"
    os.makedirs(results_dir, exist_ok=True)
    return (
        results_dir,
        f"{results_dir}/dorado_results.xlsx",
        f"{results_dir}/dorado_checkpoint.xlsx",
    )
