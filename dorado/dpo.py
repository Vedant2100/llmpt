"""Stage 5: Offline DPO training with Shangjian's hyperparameters."""

import os

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer

from dorado.utils import clear_gpu, pipeline_warn, get_mixed_precision_kwargs


def run_dpo_training(
    exp_config: dict,
    pairs: list,
    sft_model_path: str = "coldstart_dorado",
    output_path: str = "dorado_final",
) -> str | None:
    """Run DPO training on preference pairs.

    Uses Shangjian's actual hyperparameters: β=0.1, lr=5e-7, cosine schedule,
    warmup 0.1, max_grad_norm=3.0, cutoff_len=2048.

    Returns the path to the saved model/adapter, or *None* on failure.
    """
    BASE = exp_config["base_model"]
    finetuning_type = exp_config.get("finetuning_type", "lora")

    MATH_PROMPT = (
        "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
        "{q}"
    )

    # ── build DPO dataset ────────────────────────────────────────────
    total_pairs = len(pairs)
    dpo_list = [
        {"prompt": MATH_PROMPT.format(q=q), "chosen": c, "rejected": r}
        for q, c, r in pairs
        if c and r
    ]
    filtered_out = total_pairs - len(dpo_list)
    if filtered_out > 0:
        pipeline_warn(
            f"DPO: {filtered_out}/{total_pairs} pairs dropped (empty chosen/rejected)."
        )
    if not dpo_list:
        pipeline_warn("DPO: zero valid pairs after filtering. Skipping DPO training.")
        return None

    dpo_ds = datasets.Dataset.from_list(dpo_list)

    # ── tokenizer ────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )

    # ── DPO config (Shangjian's hyperparameters) ─────────────────────
    mp_kwargs = get_mixed_precision_kwargs()
    dpo_args = DPOConfig(
        output_dir=output_path,
        per_device_train_batch_size=exp_config.get("dpo_batch_size", 4),
        gradient_accumulation_steps=exp_config.get("gradient_accumulation_steps", 4),
        num_train_epochs=exp_config.get("dpo_epochs", 1),
        beta=exp_config.get("dpo_beta", 0.1),
        learning_rate=exp_config.get("dpo_lr", 5e-7),
        lr_scheduler_type=exp_config.get("dpo_lr_scheduler", "cosine"),
        warmup_ratio=exp_config.get("dpo_warmup_ratio", 0.1),
        max_grad_norm=exp_config.get("dpo_max_grad_norm", 3.0),
        max_length=exp_config.get("dpo_max_length", 2048),
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **mp_kwargs,
        remove_unused_columns=False,
        seed=exp_config.get("random_seed", 42),
    )

    # ── load model ───────────────────────────────────────────────────
    from dorado.config import make_model_load_kwargs

    load_kwargs = make_model_load_kwargs(exp_config)
    print(f"Loading base model for DPO (finetuning_type={finetuning_type})...")

    if finetuning_type == "full":
        # Full fine-tuning: load SFT model directly (or base + merge adapter)
        if os.path.exists(sft_model_path):
            # Check if SFT output is an adapter or a full model
            is_adapter = os.path.exists(
                os.path.join(sft_model_path, "adapter_config.json")
            )
            if is_adapter:
                print("Loading base + merging SFT adapter for full DPO...")
                model = AutoModelForCausalLM.from_pretrained(BASE, **load_kwargs)
                model = PeftModel.from_pretrained(model, sft_model_path)
                model = model.merge_and_unload()
            else:
                print("Loading full SFT model for DPO...")
                model = AutoModelForCausalLM.from_pretrained(
                    sft_model_path, **load_kwargs
                )
        else:
            pipeline_warn(
                f"DPO: SFT model not found at '{sft_model_path}'. Using base model."
            )
            model = AutoModelForCausalLM.from_pretrained(BASE, **load_kwargs)

        model.config.pad_token_id = tok.pad_token_id
        peft_config = None  # No LoRA for full fine-tuning

    else:
        # LoRA fine-tuning: load base, merge SFT adapter, apply new LoRA
        model = AutoModelForCausalLM.from_pretrained(BASE, **load_kwargs)

        if os.path.exists(sft_model_path):
            print("Loading SFT adapter and merging into base for DPO...")
            model = PeftModel.from_pretrained(model, sft_model_path)
            model = model.merge_and_unload()
        else:
            pipeline_warn(
                f"DPO: SFT adapter not found at '{sft_model_path}'. Using base model."
            )

        model.config.pad_token_id = tok.pad_token_id
        if hasattr(model, "peft_config"):
            delattr(model, "peft_config")

        peft_config = LoraConfig(
            r=exp_config.get("lora_r", 16),
            lora_alpha=exp_config.get("lora_alpha", 32),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    # ── train ────────────────────────────────────────────────────────
    print(f"Initializing DPOTrainer ({len(dpo_list)} pairs)...")
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=dpo_ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_path)

    del model, trainer
    clear_gpu()
    print(f"✅ DPO complete. Saved to '{output_path}'")
    return output_path
