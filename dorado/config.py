"""Experiment configuration and grid builder."""

import os
import itertools
from datetime import datetime

# ── Memory optimisation ──────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# NOTE: CUDA_VISIBLE_DEVICES must be set in the notebook BEFORE torch import

# ── Parameter dictionaries (edit these lists to sweep) ───────────────
DATASET_CONFIG = {
    "sft_samples": [500],
    "dpo_pairs": [50],
    "candidates_per_question": [4],
    "sft_dataset_name": ["tatsu-lab/alpaca"],
    "eval_split": ["test"],
    "eval_max_samples": [200],
    "random_seed": [42],
}

MODEL_CONFIG = {
    "base_model": ["Qwen/Qwen3-4B"],
    "rm_base_model": ["Qwen/Qwen3-4B"],
}

ARCHITECTURE_CONFIG = {
    "lora_r": [8],
    "lora_alpha": [16],
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


def make_model_load_kwargs(exp_config: dict, num_labels: int | None = None) -> dict:
    """Build consistent HF model loading kwargs with optional multi-GPU sharding.

    If multiple GPUs are visible via ``CUDA_VISIBLE_DEVICES``, adds a per-GPU
    ``max_memory`` cap to encourage model sharding across devices.
    """
    import torch

    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if num_labels is not None:
        load_kwargs["num_labels"] = num_labels

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        load_kwargs["token"] = hf_token

    bnb_config = make_bnb_config(exp_config)
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config

    visible = [
        d.strip()
        for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if d.strip()
    ]
    if len(visible) > 1:
        per_gpu_cap = os.environ.get("DORADO_MAX_MEMORY_PER_GPU", "10GiB")
        load_kwargs["max_memory"] = {i: per_gpu_cap for i in range(len(visible))}

    return load_kwargs


TRAINING_CONFIG = {
    "iterative_dpo_rounds": [1],
    "sft_epochs": [2],
    "rm_epochs": [3],
    "dpo_epochs": [3],
    "sft_batch_size": [2],
    "rm_batch_size": [2],
    "dpo_batch_size": [1],
}

GENERATION_CONFIG = {
    "temperature": [0.7],
    "max_new_tokens_gen": [256],
    "max_new_tokens_eval": [400],
}

EVAL_CONFIG = {
    "eval_batch_size": [2],
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
