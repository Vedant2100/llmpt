"""Stage 2: Candidate generation from a (possibly adapter-augmented) model."""

import os

from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from dorado.utils import clear_gpu


def run_candidate_generation(
    exp_config: dict,
    generator_model_path: str = "coldstart_dorado",
) -> tuple[dict, dict, list]:
    """Generate multiple candidate answers per GSM8K question.

    Returns ``(all_samples, ground_truth_map, questions)``.
    """
    BASE = exp_config["base_model"]

    math_ds = load_dataset(
        "openai/gsm8k", "main", split=f"train[:{exp_config['dpo_pairs']}]"
    )
    QUESTIONS = [x["question"] for x in math_ds]
    GT = {x["question"]: x["answer"].split("#### ")[-1].strip() for x in math_ds}

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb_config, device_map="auto"
    )
    if os.path.exists(generator_model_path):
        model = PeftModel.from_pretrained(model, generator_model_path)
    else:
        print(f"⚠️ Adapter not found at {generator_model_path}. Using base model.")

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token

    ALL_SAMPLES: dict[str, list[str]] = {}
    for q in tqdm(QUESTIONS, desc="Generating Candidates"):
        inputs = tok(q, return_tensors="pt").to(model.device)
        outs = model.generate(
            **inputs,
            max_new_tokens=exp_config["max_new_tokens_gen"],
            num_return_sequences=exp_config["candidates_per_question"],
            do_sample=True,
            temperature=exp_config["temperature"],
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        ALL_SAMPLES[q] = [
            tok.decode(o[inputs.input_ids.shape[-1] :], skip_special_tokens=True)
            for o in outs
        ]

    del model
    clear_gpu()
    print(
        f"✅ Generated {len(ALL_SAMPLES)} questions "
        f"× {exp_config['candidates_per_question']} candidates."
    )
    return ALL_SAMPLES, GT, QUESTIONS
