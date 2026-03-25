"""Stage 2: Candidate generation for self-improvement on MATH data."""

import os
import sys
import json

from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

from dorado.utils import clear_gpu, pipeline_warn


def _load_math_prompts(exp_config: dict) -> tuple[list[str], dict[str, str]]:
    """Load math prompts and ground truth from eval/data/math/test.jsonl.

    Returns (questions, ground_truth_map).
    """
    eval_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference", "eval")
    math_data_file = os.path.join(eval_dir, "data", "math", "test.jsonl")

    if not os.path.exists(math_data_file):
        raise FileNotFoundError(f"Original MATH data not found at {math_data_file}")

    questions = []
    gt_map = {}
    with open(math_data_file, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            q = d.get("problem", d.get("question", ""))
            a = d.get("answer", d.get("solution", ""))
            if q:
                questions.append(q)
                gt_map[q] = a

    count = exp_config.get("math_prompt_count", 500)
    if count and count < len(questions):
        questions = questions[:count]
        gt_map = {q: gt_map[q] for q in questions}

    print(f"📦 Loaded {len(questions)} MATH prompts from {math_data_file}")
    return questions, gt_map

    return questions, gt_map


def run_candidate_generation(
    exp_config: dict,
    generator_model_path: str = "coldstart_dorado",
) -> tuple[dict, dict, list]:
    """Generate multiple candidate answers per MATH question.

    Uses the instruction format: "Please reason step by step, and put your
    final answer within \\boxed{}."

    Returns ``(all_samples, ground_truth_map, questions)``.
    """
    BASE = exp_config["base_model"]

    QUESTIONS, GT = _load_math_prompts(exp_config)

    # ── load model ───────────────────────────────────────────────────
    from dorado.config import make_model_load_kwargs

    load_kwargs = make_model_load_kwargs(exp_config)
    model = AutoModelForCausalLM.from_pretrained(BASE, **load_kwargs)

    is_adapter = os.path.exists(
        os.path.join(generator_model_path, "adapter_config.json")
    )
    if os.path.exists(generator_model_path):
        if is_adapter:
            model = PeftModel.from_pretrained(model, generator_model_path)
        else:
            del model
            clear_gpu()
            model = AutoModelForCausalLM.from_pretrained(
                generator_model_path, **load_kwargs
            )
    else:
        pipeline_warn(
            f"Generation: model not found at '{generator_model_path}'. Using base."
        )

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

    SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

    # ── generate candidates ──────────────────────────────────────────
    ALL_SAMPLES: dict[str, list[str]] = {}
    for q in tqdm(QUESTIONS, desc="Generating Candidates"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        outs = model.generate(
            **inputs,
            max_new_tokens=exp_config.get("max_new_tokens_gen", 2048),
            num_return_sequences=exp_config["candidates_per_question"],
            do_sample=True,
            temperature=exp_config["temperature"],
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        responses = [
            tok.decode(o[inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            for o in outs
        ]
        ALL_SAMPLES[q] = responses

    # ── diagnostics ──────────────────────────────────────────────────
    empty_count = 0
    no_answer_count = 0
    total_responses = 0
    for q, responses in ALL_SAMPLES.items():
        for r in responses:
            total_responses += 1
            if not r.strip():
                empty_count += 1
            if "\\boxed" not in r and "####" not in r:
                no_answer_count += 1

    if empty_count > 0:
        pipeline_warn(
            f"Generation: {empty_count}/{total_responses} responses are empty."
        )
    if no_answer_count > 0:
        ratio = no_answer_count / max(total_responses, 1)
        pipeline_warn(
            f"Generation: {no_answer_count}/{total_responses} ({ratio:.0%}) "
            f"responses lack answer markers (\\boxed{{}} or ####)."
        )

    del model
    clear_gpu()
    print(
        f"✅ Generated {len(ALL_SAMPLES)} questions "
        f"× {exp_config['candidates_per_question']} candidates."
    )
    return ALL_SAMPLES, GT, QUESTIONS
