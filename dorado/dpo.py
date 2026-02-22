"""Stage 5: Offline DPO training."""

import os

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer

from dorado.utils import clear_gpu


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

    dpo_list = [
        {"prompt": q, "chosen": c, "rejected": r} for q, c, r in pairs if c and r
    ]
    if not dpo_list:
        print("❌ Error: No DPO pairs found!")
        return None

    dpo_ds = datasets.Dataset.from_list(dpo_list)

    dpo_args = DPOConfig(
        output_dir=output_path,
        per_device_train_batch_size=exp_config["dpo_batch_size"],
        gradient_accumulation_steps=exp_config["gradient_accumulation_steps"],
        num_train_epochs=exp_config["dpo_epochs"],
        beta=exp_config["dpo_beta"],
        logging_steps=5,
        report_to="none",
        save_strategy="no",
        fp16=True,
        remove_unused_columns=False,
    )

    # ── load & merge SFT adapter into base ───────────────────────────
    print("Loading base model + SFT adapter for DPO (8-bit)...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb_config, device_map="auto"
    )

    if os.path.exists(sft_model_path):
        print(
            "⚠️ Loading SFT adapter – will dequantize during merge (required for DPO)..."
        )
        model = PeftModel.from_pretrained(model, sft_model_path)
        model = model.merge_and_unload()
    else:
        print(f"⚠️ SFT adapter not found at {sft_model_path}. Using base model.")

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
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
