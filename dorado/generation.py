"""Stage 2: Candidate generation from a (possibly adapter-augmented) model."""

import os

from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

from dorado.utils import clear_gpu, pipeline_warn, extract_answer_from_ground_truth


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
    GT = {x["question"]: extract_answer_from_ground_truth(x["answer"]) for x in math_ds}

    from dorado.config import make_model_load_kwargs

    load_kwargs = make_model_load_kwargs(exp_config)
    model = AutoModelForCausalLM.from_pretrained(BASE, **load_kwargs)
    if os.path.exists(generator_model_path):
        model = PeftModel.from_pretrained(model, generator_model_path)
    else:
        pipeline_warn(
            f"Generation: adapter not found at '{generator_model_path}'. Using base model."
        )

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token

    MATH_PROMPT = (
        "Solve this math problem step by step. "
        "You MUST end with exactly one final line in the format: #### <number>.\n\n"
        "Question: {q}\n\nAnswer:"
    )
    RETRY_PROMPT = (
        "Solve the problem and return the final numeric answer in strict format. "
        "End with exactly one line: #### <number>.\n\n"
        "Question: {q}\n\nAnswer:"
    )

    ALL_SAMPLES: dict[str, list[str]] = {}
    for q in tqdm(QUESTIONS, desc="Generating Candidates"):
        prompt = MATH_PROMPT.format(q=q)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        outs = model.generate(
            **inputs,
            max_new_tokens=exp_config["max_new_tokens_gen"],
            num_return_sequences=exp_config["candidates_per_question"],
            do_sample=True,
            temperature=exp_config["temperature"],
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        responses = [
            tok.decode(o[inputs.input_ids.shape[-1] :], skip_special_tokens=True)
            for o in outs
        ]

        if exp_config.get("retry_failed_generations", True):
            retry_temperature = exp_config.get("retry_temperature", 0.2)
            for idx, resp in enumerate(responses):
                if "####" in resp and resp.strip():
                    continue
                retry_inputs = tok(RETRY_PROMPT.format(q=q), return_tensors="pt").to(
                    model.device
                )
                retry_out = model.generate(
                    **retry_inputs,
                    max_new_tokens=exp_config["max_new_tokens_gen"],
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=retry_temperature,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                )
                responses[idx] = tok.decode(
                    retry_out[0][retry_inputs.input_ids.shape[-1] :],
                    skip_special_tokens=True,
                )

        ALL_SAMPLES[q] = responses

    # ── post-generation diagnostics ─────────────────────────────────
    empty_count = 0
    no_marker_count = 0
    for q, responses in ALL_SAMPLES.items():
        for r in responses:
            if not r.strip():
                empty_count += 1
            if "####" not in r:
                no_marker_count += 1
    total_responses = len(ALL_SAMPLES) * exp_config["candidates_per_question"]
    if empty_count > 0:
        pipeline_warn(
            f"Generation: {empty_count}/{total_responses} candidate responses "
            f"are empty."
        )
    if no_marker_count > 0:
        ratio = no_marker_count / max(total_responses, 1)
        pipeline_warn(
            f"Generation: {no_marker_count}/{total_responses} ({ratio:.0%}) "
            f"candidates lack '####' answer marker."
        )

    del model
    clear_gpu()
    print(
        f"✅ Generated {len(ALL_SAMPLES)} questions "
        f"× {exp_config['candidates_per_question']} candidates."
    )
    return ALL_SAMPLES, GT, QUESTIONS
