"""Stage 1: Cold-Start SFT (GC-Boost) — general chat data for reasoning alignment."""

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


def _format_chat_example(example: dict, tokenizer) -> dict:
    """Format a chat example using the model's chat template.

    Handles both ultrachat_200k format (messages list) and
    generic instruction format (instruction/input/output fields).
    """
    if "messages" in example:
        # ultrachat_200k format: list of {role, content} dicts
        messages = example["messages"]
    elif "prompt" in example and "completion" in example:
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    elif "instruction" in example:
        user_content = example["instruction"]
        if example.get("input", "").strip():
            user_content += "\n" + example["input"]
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example.get("output", "")},
        ]
    else:
        raise ValueError(f"Unsupported dataset format. Keys: {list(example.keys())}")

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def run_sft_stage(exp_config: dict, output_dir: str = "coldstart_dorado") -> str:
    """Fine-tune base model on general chat data (GC-Boost stage).

    Supports both LoRA and full fine-tuning based on exp_config["finetuning_type"].
    Returns the path to the saved model/adapter directory.
    """
    BASE = exp_config["base_model"]
    finetuning_type = exp_config.get("finetuning_type", "lora")

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        # Fallback ChatML template for models without one (e.g. SmolLM2-135M)
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )

    # ── load model ───────────────────────────────────────────────────
    from dorado.config import make_model_load_kwargs

    load_kwargs = make_model_load_kwargs(exp_config)
    model = AutoModelForCausalLM.from_pretrained(BASE, **load_kwargs)

    if finetuning_type == "lora":
        if "quantization_config" in load_kwargs:
            model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=exp_config["lora_r"],
            lora_alpha=exp_config["lora_alpha"],
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.print_trainable_parameters()
    else:
        # Full fine-tuning: enable gradient for all params
        for param in model.parameters():
            param.requires_grad = True
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full fine-tuning: {trainable:,}/{total_params:,} params trainable")

    # ── load dataset (stream to save disk) ─────────────────────────
    dataset_name = exp_config.get("sft_dataset_name", "HuggingFaceH4/ultrachat_200k")
    dataset_split = exp_config.get("sft_dataset_split", "train_sft")
    sft_samples = exp_config.get("sft_samples", 5000)

    from datasets import Dataset
    streamed = load_dataset(dataset_name, split=dataset_split, streaming=True)
    rows = list(streamed.take(sft_samples))
    dataset = Dataset.from_list(rows)
    print(f"📦 SFT dataset: {dataset_name} ({len(dataset)} samples, streamed)")

    # ── tokenize with chat template ──────────────────────────────────
    cutoff_len = exp_config.get("sft_cutoff_len", 2048)

    def tokenize_fn(example):
        formatted = _format_chat_example(example, tokenizer)
        return tokenizer(
            formatted["text"],
            truncation=True,
            max_length=cutoff_len,
        )

    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        num_proc=min(os.cpu_count() or 1, 8),
        desc="Tokenizing SFT data",
    )

    # ── training args ────────────────────────────────────────────────
    mp_kwargs = get_mixed_precision_kwargs()
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=exp_config.get("sft_batch_size", 4),
        gradient_accumulation_steps=exp_config.get("gradient_accumulation_steps", 4),
        num_train_epochs=exp_config.get("sft_epochs", 3),
        learning_rate=exp_config.get("sft_lr", 2e-5),
        lr_scheduler_type=exp_config.get("sft_lr_scheduler", "cosine"),
        warmup_ratio=exp_config.get("sft_warmup_ratio", 0.1),
        logging_steps=10,
        save_strategy="no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **mp_kwargs,
        report_to="none",
        seed=exp_config.get("random_seed", 42),
    )

    # ── train ────────────────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model, trainer
    clear_gpu()
    print(f"✅ SFT (GC-Boost) complete. Saved to '{output_dir}'")
    return output_dir
