"""Experiment configuration and grid builder with profile support."""

import os
import itertools
from datetime import datetime
from copy import deepcopy

# ── Memory optimisation ──────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ── Profile definitions ──────────────────────────────────────────────

PROFILES = {
    "smoke": {
        # Model
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "finetuning_type": "lora",  # "lora" | "full"
        # SFT
        "sft_dataset_name": "HuggingFaceH4/ultrachat_200k",
        "sft_dataset_split": "train_sft",
        "sft_samples": 50,
        "sft_cutoff_len": 512,
        "sft_epochs": 1,
        "sft_lr": 2e-5,
        "sft_lr_scheduler": "cosine",
        "sft_warmup_ratio": 0.1,
        "sft_batch_size": 2,
        # LoRA (only used when finetuning_type == "lora")
        "lora_r": 8,
        "lora_alpha": 16,
        # Generation
        "math_prompt_source": "MATH",
        "math_prompt_count": 20,
        "candidates_per_question": 2,
        "temperature": 1.0,
        "max_new_tokens_gen": 512,
        # RM
        "rm_strategy": "skip",  # "skip" | "armo"
        # DPO
        "dpo_beta": 0.1,
        "dpo_lr": 5e-7,
        "dpo_lr_scheduler": "cosine",
        "dpo_warmup_ratio": 0.1,
        "dpo_epochs": 1,
        "dpo_batch_size": 2,
        "dpo_max_length": 512,
        "dpo_max_grad_norm": 3.0,
        "gradient_accumulation_steps": 1,
        # Eval
        "eval_benchmarks": ["math"],
        "eval_max_samples": 50,
        "eval_engine": "hf",  # "hf" | "vllm"
        "max_new_tokens_eval": 1024,
        "eval_batch_size": 2,
        # General
        "random_seed": 42,
        "quantization_bits": 0,
        "deepspeed_config": None,
        "iterative_dpo_rounds": 1,
    },
    "fast": {
        # Model
        "base_model": "Qwen/Qwen2.5-Math-1.5B",
        "finetuning_type": "lora",
        # SFT
        "sft_dataset_name": "HuggingFaceH4/ultrachat_200k",
        "sft_dataset_split": "train_sft",
        "sft_samples": 1000,
        "sft_cutoff_len": 2048,
        "sft_epochs": 3,
        "sft_lr": 2e-5,
        "sft_lr_scheduler": "cosine",
        "sft_warmup_ratio": 0.1,
        "sft_batch_size": 4,
        # LoRA
        "lora_r": 16,
        "lora_alpha": 32,
        # Generation
        "math_prompt_source": "MATH",
        "math_prompt_count": 500,
        "candidates_per_question": 5,
        "temperature": 1.0,
        "max_new_tokens_gen": 2048,
        # RM
        "rm_strategy": "armo",
        # DPO
        "dpo_beta": 0.1,
        "dpo_lr": 5e-7,
        "dpo_lr_scheduler": "cosine",
        "dpo_warmup_ratio": 0.1,
        "dpo_epochs": 1,
        "dpo_batch_size": 4,
        "dpo_max_length": 2048,
        "dpo_max_grad_norm": 3.0,
        "gradient_accumulation_steps": 4,
        # Eval
        "eval_benchmarks": ["math"],
        "eval_max_samples": 200,
        "eval_engine": "vllm",
        "max_new_tokens_eval": 2048,
        "eval_batch_size": 4,
        # General
        "random_seed": 42,
        "quantization_bits": 0,
        "deepspeed_config": None,
        "iterative_dpo_rounds": 1,
    },
    "full": {
        # Model
        "base_model": "Qwen/Qwen2.5-Math-7B",
        "finetuning_type": "full",
        # SFT
        "sft_dataset_name": "HuggingFaceH4/ultrachat_200k",
        "sft_dataset_split": "train_sft",
        "sft_samples": 10000,
        "sft_cutoff_len": 8192,
        "sft_epochs": 3,
        "sft_lr": 2e-5,
        "sft_lr_scheduler": "cosine",
        "sft_warmup_ratio": 0.1,
        "sft_batch_size": 4,
        # LoRA (not used in full profile, but kept for override flexibility)
        "lora_r": 16,
        "lora_alpha": 32,
        # Generation
        "math_prompt_source": "MATH",
        "math_prompt_count": 8000,
        "candidates_per_question": 5,
        "temperature": 1.0,
        "max_new_tokens_gen": 2048,
        # RM
        "rm_strategy": "armo",
        # DPO
        "dpo_beta": 0.1,
        "dpo_lr": 5e-7,
        "dpo_lr_scheduler": "cosine",
        "dpo_warmup_ratio": 0.1,
        "dpo_epochs": 1,
        "dpo_batch_size": 4,
        "dpo_max_length": 2048,
        "dpo_max_grad_norm": 3.0,
        "gradient_accumulation_steps": 4,
        # Eval
        "eval_benchmarks": ["math", "minerva", "amc", "aime", "olympiadbench"],
        "eval_max_samples": None,  # use full benchmark
        "eval_engine": "vllm",
        "max_new_tokens_eval": 32768,
        "eval_batch_size": 8,
        # General
        "random_seed": 42,
        "quantization_bits": 0,
        "deepspeed_config": "examples/deepspeed/ds_z3_config.json",
        "iterative_dpo_rounds": 1,
    },
}


def get_profile(name: str = "fast", overrides: dict | None = None) -> dict:
    """Return a complete experiment config for a named profile.

    Parameters
    ----------
    name : str
        One of "smoke", "fast", or "full".
    overrides : dict, optional
        Key-value pairs to override specific profile settings.
    """
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Choose from: {list(PROFILES.keys())}")
    config = deepcopy(PROFILES[name])
    config["profile"] = name
    if overrides:
        config.update(overrides)
    return config


# ── BitsAndBytes config (for quantized runs) ─────────────────────────

def make_bnb_config(exp_config: dict):
    """Build a BitsAndBytesConfig, or None if quantization is disabled."""
    import torch
    from transformers import BitsAndBytesConfig

    bits = exp_config.get("quantization_bits", 0)
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        return None


def make_model_load_kwargs(exp_config: dict, num_labels: int | None = None) -> dict:
    """Build consistent HF model loading kwargs with optional multi-GPU sharding."""
    import torch

    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
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
        per_gpu_cap = os.environ.get("DORADO_MAX_MEMORY_PER_GPU", "22GiB")
        load_kwargs["max_memory"] = {i: per_gpu_cap for i in range(len(visible))}

    return load_kwargs


# ── Grid helpers (kept for backward compatibility) ───────────────────

def build_experiment_grid(
    profile: str = "fast",
    overrides: dict | None = None,
    sweep_params: dict | None = None,
) -> list[dict]:
    """Generate experiment configs from a profile with optional param sweeps.

    Parameters
    ----------
    profile : str
        Base profile name.
    overrides : dict, optional
        Static overrides applied to every experiment.
    sweep_params : dict, optional
        Dict of {param_name: [value1, value2, ...]} for grid search.
        If None, returns a single-element list with the profile config.
    """
    base = get_profile(profile, overrides)

    if not sweep_params:
        base["experiment_id"] = 0
        return [base]

    keys = list(sweep_params.keys())
    values = [sweep_params[k] for k in keys]
    experiments = []
    for i, combo in enumerate(itertools.product(*values)):
        exp = deepcopy(base)
        for k, v in zip(keys, combo):
            exp[k] = v
        exp["experiment_id"] = i
        experiments.append(exp)
    return experiments


def estimate_time(exp: dict) -> dict:
    """Rough runtime estimate (minutes)."""
    sft_min = 0.004 * exp["sft_samples"] * exp["sft_epochs"]
    gen_min = 0.002 * exp["math_prompt_count"] * (exp["candidates_per_question"] / 2)
    dpo_min = 0.003 * exp["math_prompt_count"] * exp["dpo_epochs"]
    eval_samples = exp.get("eval_max_samples") or 500
    eval_min = 0.01 * eval_samples / exp["eval_batch_size"]
    rounds = max(1, exp["iterative_dpo_rounds"])
    total = sft_min + (gen_min + dpo_min) * rounds + eval_min
    return dict(
        sft_min=sft_min,
        gen_min=gen_min,
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
