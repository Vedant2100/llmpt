"""Stage 4: Reward-model training on preference pairs."""

import datasets
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from dorado.utils import clear_gpu, pipeline_warn, get_mixed_precision_kwargs


def run_rm_training(
    exp_config: dict,
    pairs: list,
    labels: list,
    output_dir: str = "reward_model",
) -> str:
    """Train a binary reward model on (good, bad) pairs.

    Returns the path to the saved adapter directory.
    """
    BASE = exp_config["rm_base_model"]

    if not pairs:
        pipeline_warn("Reward model: received zero pairs. Skipping RM training.")
        return output_dir

    # ── build dataset ────────────────────────────────────────────────
    data = []
    for (q, good, bad), _lab in tqdm(
        zip(pairs, labels), total=len(labels), desc="Build reward data"
    ):
        data.append({"text": q + " [ANS] " + good, "label": 1})
        data.append({"text": q + " [ANS] " + bad, "label": 0})

    if len(data) < 6:
        pipeline_warn(
            f"Reward model: only {len(data)} samples (<6). "
            f"Using same data for train and test."
        )
        full_ds = datasets.Dataset.from_list(data)
        ds = datasets.DatasetDict({"train": full_ds, "test": full_ds})
    else:
        ds = datasets.Dataset.from_list(data).train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.pad_token = tokenizer.eos_token

    # ── model ────────────────────────────────────────────────────────
    from dorado.config import make_model_load_kwargs

    load_kwargs = make_model_load_kwargs(exp_config, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, **load_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    if "quantization_config" in load_kwargs:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=exp_config["lora_r"],
        lora_alpha=exp_config["lora_alpha"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def tok_fn(ex):
        return tokenizer(
            ex["text"], truncation=True, padding="max_length", max_length=256
        )

    tok_ds = ds.map(tok_fn, batched=True, remove_columns=["text"])

    # ── train ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            save_strategy="no",
            per_device_train_batch_size=exp_config["rm_batch_size"],
            num_train_epochs=exp_config["rm_epochs"],
            logging_steps=10,
            **get_mixed_precision_kwargs(),
            report_to="none",
        ),
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["test"],
    )
    trainer.train()
    trainer.save_model(output_dir)

    del model, trainer
    clear_gpu()
    print("✅ Reward model training complete.")
    return output_dir
