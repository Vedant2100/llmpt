"""Stage 5: Offline DPO training."""

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

    Returns the path to the saved adapter, or *None* on failure.
    """
    BASE = exp_config["base_model"]

    MATH_PROMPT = (
        "Solve this math problem step by step. Put your final numeric answer "
        "after ####.\n\nQuestion: {q}\n\nAnswer:"
    )

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

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token

    prompt_max_tokens = 256
    filtered_dpo_list = []
    overlong_count = 0
    for row in dpo_list:
        prompt_len = len(tok(row["prompt"], add_special_tokens=False)["input_ids"])
        if prompt_len <= prompt_max_tokens:
            filtered_dpo_list.append(row)
        else:
            overlong_count += 1

    if overlong_count > 0:
        pipeline_warn(
            f"DPO: {overlong_count}/{len(dpo_list)} pairs dropped (prompt too long > {prompt_max_tokens} tokens)."
        )
    if not filtered_dpo_list:
        pipeline_warn(
            "DPO: zero valid pairs after prompt-length filtering. Skipping DPO training."
        )
        return None

    dpo_ds = datasets.Dataset.from_list(filtered_dpo_list)

    dpo_args = DPOConfig(
        output_dir=output_path,
        per_device_train_batch_size=exp_config["dpo_batch_size"],
        gradient_accumulation_steps=exp_config["gradient_accumulation_steps"],
        num_train_epochs=exp_config["dpo_epochs"],
        beta=exp_config["dpo_beta"],
        logging_steps=5,
        report_to="none",
        save_strategy="no",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=512,
        **get_mixed_precision_kwargs(),
        remove_unused_columns=False,
    )

    # ── load & merge SFT adapter into base ───────────────────────────
    bits = exp_config.get("quantization_bits", 0)
    print(f"Loading base model + SFT adapter for DPO ({bits}-bit if >0, else fp16)...")
    from dorado.config import make_model_load_kwargs

    load_kwargs = make_model_load_kwargs(exp_config)
    model = AutoModelForCausalLM.from_pretrained(BASE, **load_kwargs)

    if os.path.exists(sft_model_path):
        print(
            "Loading SFT adapter – will dequantize during merge (required for DPO)..."
        )
        model = PeftModel.from_pretrained(model, sft_model_path)
        model = model.merge_and_unload()
    else:
        pipeline_warn(
            f"DPO: SFT adapter not found at '{sft_model_path}'. Using base model."
        )

    model.config.pad_token_id = tok.pad_token_id

    peft_config = LoraConfig(
        r=exp_config["lora_r"],
        lora_alpha=exp_config["lora_alpha"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    # ── train ────────────────────────────────────────────────────────
    print("Initializing DPOTrainer...")
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
    print(f"✅ DPO Complete. '{output_path}' saved.")
    return output_path
