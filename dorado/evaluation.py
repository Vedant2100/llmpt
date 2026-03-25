"""Stage 6: Evaluation on MATH benchmarks with proper answer parsing."""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from math import comb

from dorado.utils import clear_gpu, pipeline_warn


# ── eval utils integration ───────────────────────────────────────────

def _setup_eval_imports():
    """Add eval/ directory to sys.path and import answer checking utils."""
    eval_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference", "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    try:
        from utils.parser import extract_answer, parse_question, parse_ground_truth, strip_string
        from utils.grader import check_is_correct
        return extract_answer, parse_question, parse_ground_truth, check_is_correct, strip_string
    except ImportError as e:
        pipeline_warn(f"Could not import eval utilities: {e}")
        return None, None, None, None, None


# ── data loading ─────────────────────────────────────────────────────

def _load_benchmark(benchmark_name: str, max_samples: int | None = None) -> list[dict]:
    """Load benchmark data from eval/data/{name}/test.jsonl."""
    eval_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference", "eval")
    data_file = os.path.join(eval_dir, "data", benchmark_name, "test.jsonl")

    if not os.path.exists(data_file):
        pipeline_warn(f"Benchmark data not found: {data_file}")
        return []

    examples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if max_samples and max_samples < len(examples):
        examples = examples[:max_samples]

    print(f"📦 Loaded {len(examples)} examples from {benchmark_name}")
    return examples


# ── helpers ──────────────────────────────────────────────────────────

def _batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def bootstrap_confidence_interval(
    correct_flags, n_bootstrap: int = 1000, confidence: float = 0.95
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for accuracy."""
    arr = np.asarray(correct_flags, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    accs = [
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ]
    alpha = 1 - confidence
    return (
        float(np.percentile(accs, (alpha / 2) * 100)),
        float(np.percentile(accs, (1 - alpha / 2) * 100)),
    )


def compute_pass_at_k(is_correct_list: list[bool], k: int) -> float:
    """Unbiased pass@k estimator."""
    n = len(is_correct_list)
    c = sum(is_correct_list)
    if n == 0:
        return 0.0
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (comb(n - c, k) / comb(n, k))


# ── HF generate evaluation ──────────────────────────────────────────

def _evaluate_hf(
    model_path: str,
    exp_config: dict,
    examples: list[dict],
    benchmark_name: str,
) -> dict:
    """Evaluate using HuggingFace model.generate()."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from dorado.config import make_model_load_kwargs

    BASE = exp_config["base_model"]
    extract_answer, parse_question_fn, parse_ground_truth_fn, check_is_correct, strip_string = (
        _setup_eval_imports()
    )

    if extract_answer is None:
        pipeline_warn("eval utilities not available. Cannot evaluate.")
        return {}

    # Load model
    load_kwargs = make_model_load_kwargs(exp_config)
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_adapter:
        print(f"Loading as PEFT adapter from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(BASE, **load_kwargs)
        model = PeftModel.from_pretrained(model, model_path)
    elif os.path.exists(model_path) and model_path != BASE:
        print(f"Loading full model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    else:
        print(f"Loading base model {BASE}...")
        model = AutoModelForCausalLM.from_pretrained(BASE, **load_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id

    SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

    correct_cnt = 0
    total = len(examples)
    results = []

    for batch in tqdm(
        list(_batched(examples, exp_config.get("eval_batch_size", 4))),
        desc=f"Eval {benchmark_name}",
    ):
        prompts = []
        for ex in batch:
            question = parse_question_fn(ex, benchmark_name)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
            model.device
        )
        with torch.no_grad():
            outs = model.generate(
                **enc,
                max_new_tokens=exp_config.get("max_new_tokens_eval", 2048),
                do_sample=False,
            )

        for i, ex in enumerate(batch):
            response = tokenizer.decode(
                outs[i][enc.input_ids.shape[1]:], skip_special_tokens=True
            )
            question = parse_question_fn(ex, benchmark_name)
            _, gt_answer = parse_ground_truth_fn(ex, benchmark_name)

            predicted = extract_answer(response)
            is_correct = check_is_correct(predicted, gt_answer) if gt_answer else False

            if is_correct:
                correct_cnt += 1

            results.append({
                "question": question,
                "gold_answer": gt_answer,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "response": response[:500],
            })

    accuracy = correct_cnt / total if total > 0 else 0.0
    correct_flags = [r["is_correct"] for r in results]
    ci_lo, ci_hi = bootstrap_confidence_interval(correct_flags)

    del model
    clear_gpu()

    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct_count": correct_cnt,
        "total_count": total,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "results": results,
    }


# ── vLLM evaluation ─────────────────────────────────────────────────

def _evaluate_vllm(
    model_path: str,
    exp_config: dict,
    examples: list[dict],
    benchmark_name: str,
) -> dict:
    """Evaluate using vLLM for fast batched inference."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        pipeline_warn("vLLM not installed. Falling back to HF generate.")
        return _evaluate_hf(model_path, exp_config, examples, benchmark_name)

    from transformers import AutoTokenizer

    extract_answer, parse_question_fn, parse_ground_truth_fn, check_is_correct, strip_string = (
        _setup_eval_imports()
    )
    if extract_answer is None:
        pipeline_warn("eval utilities not available. Cannot evaluate.")
        return {}

    BASE = exp_config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )

    SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

    # Build prompts
    prompts = []
    for ex in examples:
        question = parse_question_fn(ex, benchmark_name)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)

    # Determine which model to load
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if is_adapter:
        # vLLM doesn't natively support adapter-only; merge first
        pipeline_warn("vLLM eval: adapter detected. Merging adapter before vLLM inference...")
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        from dorado.config import make_model_load_kwargs

        load_kwargs = make_model_load_kwargs(exp_config)
        model = AutoModelForCausalLM.from_pretrained(BASE, **load_kwargs)
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()

        merged_path = model_path + "_merged"
        model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        del model
        clear_gpu()
        vllm_model_path = merged_path
    else:
        vllm_model_path = model_path if os.path.exists(model_path) else BASE

    # Run vLLM
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    tp_size = len([g for g in available_gpus if g.strip()])

    llm = LLM(
        model=vllm_model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=exp_config.get("max_new_tokens_eval", 2048),
    )

    print(f"vLLM inference on {len(prompts)} prompts (TP={tp_size})...")
    completions = llm.generate(prompts, sampling_params)

    # Grade
    correct_cnt = 0
    results = []
    for i, ex in enumerate(examples):
        response = completions[i].outputs[0].text
        question = parse_question_fn(ex, benchmark_name)
        _, gt_answer = parse_ground_truth_fn(ex, benchmark_name)

        predicted = extract_answer(response)
        is_correct = check_is_correct(predicted, gt_answer) if gt_answer else False

        if is_correct:
            correct_cnt += 1

        results.append({
            "question": question,
            "gold_answer": gt_answer,
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "response": response[:500],
        })

    total = len(examples)
    accuracy = correct_cnt / total if total > 0 else 0.0
    correct_flags = [r["is_correct"] for r in results]
    ci_lo, ci_hi = bootstrap_confidence_interval(correct_flags)

    del llm
    clear_gpu()

    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct_count": correct_cnt,
        "total_count": total,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "results": results,
    }


# ── public API ───────────────────────────────────────────────────────

def evaluate_model(
    exp_config: dict,
    model_path: str,
    model_label: str,
) -> dict:
    """Evaluate a model across all configured benchmarks.

    Returns a dict with per-benchmark metrics and aggregated results.
    """
    benchmarks = exp_config.get("eval_benchmarks", ["math"])
    max_samples = exp_config.get("eval_max_samples")
    eval_engine = exp_config.get("eval_engine", "hf")

    print(f"\n{'─'*60}")
    print(f"Evaluating {model_label} on {benchmarks} (engine={eval_engine})")
    print(f"{'─'*60}")

    all_benchmark_results = {}
    total_correct = 0
    total_count = 0

    for bench in benchmarks:
        examples = _load_benchmark(bench, max_samples)
        if not examples:
            continue

        eval_fn = _evaluate_vllm if eval_engine == "vllm" else _evaluate_hf
        try:
            metrics = eval_fn(model_path, exp_config, examples, bench)
        except Exception as e:
            pipeline_warn(f"Eval failed for {model_label} on {bench}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if metrics:
            all_benchmark_results[bench] = metrics
            total_correct += metrics.get("correct_count", 0)
            total_count += metrics.get("total_count", 0)
            print(
                f"  {bench}: {metrics['accuracy']:.1%} "
                f"({metrics['correct_count']}/{metrics['total_count']}) "
                f"CI=[{metrics['ci_lower']:.1%}, {metrics['ci_upper']:.1%}]"
            )

    avg_accuracy = total_correct / total_count if total_count > 0 else 0.0
    print(f"\n  📊 {model_label} average: {avg_accuracy:.1%} ({total_correct}/{total_count})")

    return {
        "model_label": model_label,
        "avg_accuracy": avg_accuracy,
        "total_correct": total_correct,
        "total_count": total_count,
        "per_benchmark": all_benchmark_results,
    }


def run_full_evaluation(
    exp_config: dict,
    model_paths_dict: dict[str, str],
) -> tuple[dict, list]:
    """Evaluate all models, print summary.

    Returns ``(all_metrics, all_results)``.
    """
    all_metrics: dict[str, dict] = {}
    all_results: list[dict] = []

    for name, path in model_paths_dict.items():
        m = evaluate_model(exp_config, path, name)
        if m:
            all_metrics[name] = m
            for bench_data in m.get("per_benchmark", {}).values():
                for r in bench_data.get("results", []):
                    r["Model"] = name
                    all_results.append(r)

    # ── summary table ────────────────────────────────────────────────
    if all_metrics:
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        rows = []
        for name, m in all_metrics.items():
            row = {"Model": name, "Avg Accuracy %": m["avg_accuracy"] * 100}
            for bench, bm in m.get("per_benchmark", {}).items():
                row[f"{bench} %"] = bm["accuracy"] * 100
            rows.append(row)
        summary_df = pd.DataFrame(rows)
        print(summary_df.to_string(index=False))

    return all_metrics, all_results
