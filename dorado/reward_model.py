"""Stage 4: Reward-model training on preference pairs."""

import datasets
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from dorado.utils import clear_gpu


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

    # ── build dataset ────────────────────────────────────────────────
    data = []
    for (q, good, bad), _lab in tqdm(
        zip(pairs, labels), total=len(labels), desc="Build reward data"
    ):
        data.append({"text": q + " [ANS] " + good, "label": 1})
        data.append({"text": q + " [ANS] " + bad, "label": 0})

    ds = datasets.Dataset.from_list(data).train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.pad_token = tokenizer.eos_token

    # ── model ────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=2, quantization_config=bnb_config, device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
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
            ex["text"], truncation=True, padding="max_length", max_length=512
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
            fp16=True,
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
