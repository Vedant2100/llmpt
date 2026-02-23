"""Stage 1: Cold-Start SFT training."""

import os

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from dorado.utils import clear_gpu, get_mixed_precision_kwargs


def run_sft_stage(exp_config: dict, output_dir: str = "coldstart_dorado") -> str:
    """Fine-tune base model on instruction data with LoRA (8-bit).

    Returns the path to the saved adapter directory.
    """
    BASE = exp_config["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.pad_token = tokenizer.eos_token

    from dorado.config import make_model_load_kwargs

    load_kwargs = make_model_load_kwargs(exp_config)
    model = AutoModelForCausalLM.from_pretrained(BASE, **load_kwargs)
    if "quantization_config" in load_kwargs:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=exp_config["lora_r"],
        lora_alpha=exp_config["lora_alpha"],
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset(
        exp_config["sft_dataset_name"],
        split=f"train[:{exp_config['sft_samples']}]",
    )

    def tok_fn(ex):
        prompt = (
            f"Instruction: {ex['instruction']}\n"
            f"Input: {ex.get('input', '')}\n"
            f"Response: {ex['output']}"
        )
        return tokenizer(prompt, truncation=True, max_length=512)

    tokenized = dataset.map(tok_fn, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=exp_config["sft_batch_size"],
        gradient_accumulation_steps=exp_config["gradient_accumulation_steps"],
        num_train_epochs=exp_config["sft_epochs"],
        logging_steps=10,
        save_strategy="no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **get_mixed_precision_kwargs(),
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized, data_collator=data_collator
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model, trainer
    clear_gpu()
    print(f"✅ SFT Complete. '{output_dir}' saved.")
    return output_dir
