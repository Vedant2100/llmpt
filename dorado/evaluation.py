"""Stage 6: Evaluation with bootstrap CIs and McNemar's test."""

import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from dorado.utils import clear_gpu, extract_answer_from_response, pipeline_warn


# ── helpers ──────────────────────────────────────────────────────────


def _batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def bootstrap_confidence_interval(
    correct_flags, n_bootstrap: int = 1000, confidence: float = 0.95
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for accuracy."""
    arr = np.asarray(correct_flags)
    accs = [
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = 1 - confidence
    return (
        float(np.percentile(accs, (alpha / 2) * 100)),
        float(np.percentile(accs, (1 - alpha / 2) * 100)),
    )


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts.

    Uses SciPy's binomtest when available and falls back to a pure-Python
    binomial tail computation when needed.
    """
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)
    try:
        from scipy.stats import binomtest

        return float(binomtest(k, n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        from math import comb

        cdf = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
        return float(min(1.0, 2.0 * cdf))


# ── single-model evaluation ─────────────────────────────────────────


def evaluate_model(
    exp_config: dict,
    model_path: str,
    model_label: str,
    prompts: list[str],
    gt: dict[str, str],
) -> dict | None:
    """Evaluate one model; return metrics dict or *None* on skip/error."""
    BASE = exp_config["base_model"]
    SFT_OUT = "coldstart_dorado"

    if not os.path.exists(model_path) and model_label != "BASE":
        pipeline_warn(f"Eval: {model_label} path '{model_path}' not found. Skipping.")
        return None

    print(f"Evaluating {model_label}...")
    results: list[dict] = []
    correct_flags: list[bool] = []
    total_response_length = 0
    parsed_answer_count = 0

    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    try:
        from dorado.config import make_model_load_kwargs

        bits = exp_config.get("quantization_bits", 0)
        load_kwargs = make_model_load_kwargs(exp_config)
        label = f"{bits}-bit" if bits > 0 else "fp16"
        if is_adapter:
            print(f"Loading {model_label} as PEFT adapter ({label})...")
            model = AutoModelForCausalLM.from_pretrained(BASE, **load_kwargs)
            is_dorado_variant = model_label.startswith("DORADO")
            if is_dorado_variant:
                if os.path.exists(SFT_OUT):
                    print("  Stacking SFT adapter...")
                    model = PeftModel.from_pretrained(model, SFT_OUT)
                print(f"  Stacking {model_label} adapter...")
                model = PeftModel.from_pretrained(model, model_path)
            else:
                model = PeftModel.from_pretrained(model, model_path)
        else:
            print(f"Loading {model_label} as full model ({label})...")
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(BASE)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        MATH_PROMPT = "Solve this math problem step by step. Put your final numeric answer after ####.\n\nQuestion: {q}\n\nAnswer:"

        for batch_prompts in tqdm(
            list(_batched(prompts, exp_config["eval_batch_size"])),
            desc="Eval batches",
        ):
            formatted = [MATH_PROMPT.format(q=p) for p in batch_prompts]
            enc = tokenizer(formatted, return_tensors="pt", padding=True).to(
                model.device
            )
            outs = model.generate(
                **enc, max_new_tokens=exp_config["max_new_tokens_eval"]
            )
            for i, prompt in enumerate(batch_prompts):
                resp = tokenizer.decode(
                    outs[i][enc.input_ids.shape[1] :], skip_special_tokens=True
                )
                predicted = extract_answer_from_response(resp)
                ground_truth = gt.get(prompt, "?")
                ok = predicted == ground_truth

                correct_flags.append(ok)
                if predicted != "None":
                    parsed_answer_count += 1
                total_response_length += len(resp.split())

                results.append(
                    {
                        "Model": model_label,
                        "Prompt": prompt,
                        "Correct Answer": ground_truth,
                        "Model Answer": predicted,
                        "Accurate": "✅" if ok else "❌",
                        "Full Response": resp.strip(),
                    }
                )

        accuracy = float(np.mean(correct_flags))
        avg_len = total_response_length / len(prompts)
        parsed_ratio = parsed_answer_count / len(prompts)
        ci_lo, ci_hi = bootstrap_confidence_interval(correct_flags)

        if parsed_ratio < 0.5:
            pipeline_warn(
                f"Eval: {model_label} parsed ratio is {parsed_ratio:.0%} (<50%). "
                f"Model may not be following the #### answer format."
            )
        if accuracy == 0.0:
            pipeline_warn(
                f"Eval: {model_label} has 0% accuracy. Model is not solving any problems."
            )

        print(
            f"✨ {model_label} – Accuracy: {accuracy:.1%} "
            f"(95% CI: [{ci_lo:.1%}, {ci_hi:.1%}])"
        )
        print(f"   Avg Length: {avg_len:.1f} words, Parsed: {parsed_ratio:.1%}")

        del model
        clear_gpu()

        return {
            "model_label": model_label,
            "accuracy": accuracy,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "ci_width": ci_hi - ci_lo,
            "correct_count": sum(correct_flags),
            "total_count": len(prompts),
            "avg_response_length": avg_len,
            "parsed_answer_ratio": parsed_ratio,
            "correct_flags": correct_flags,
            "results": results,
        }
    except Exception as e:
        print(f"❌ Error evaluating {model_label}: {e}")
        try:
            del model
            clear_gpu()
        except NameError:
            pass
        return None


# ── multi-model evaluation ───────────────────────────────────────────


def run_full_evaluation(
    exp_config: dict,
    model_paths_dict: dict[str, str],
) -> tuple[dict, list]:
    """Evaluate all models, print summary & significance test.

    Returns ``(all_metrics, all_results)``.
    """
    eval_ds = load_dataset("openai/gsm8k", "main", split=exp_config["eval_split"])
    max_n = exp_config.get("eval_max_samples")
    if max_n is not None:
        eval_ds = eval_ds.select(range(min(max_n, len(eval_ds))))

    questions = [x["question"] for x in eval_ds]
    gt = {
        x["question"]: x["answer"].split("#### ")[-1].strip().replace(",", "")
        for x in eval_ds
    }

    all_metrics: dict[str, dict] = {}
    all_results: list[dict] = []

    for name, path in model_paths_dict.items():
        m = evaluate_model(exp_config, path, name, questions, gt)
        if m:
            all_metrics[name] = m
            all_results.extend(m["results"])

    if not all_metrics:
        pipeline_warn("Eval: all models were skipped or failed. No metrics produced.")

    # ── summary table ────────────────────────────────────────────────
    if all_results:
        print("\n--- PERFORMANCE SUMMARY ---")
        rows = [
            {
                "Model": m["model_label"],
                "Accuracy %": m["accuracy"] * 100,
                "95% CI Lower": m["ci_lower"] * 100,
                "95% CI Upper": m["ci_upper"] * 100,
                "CI Width": m["ci_width"] * 100,
                "n": m["total_count"],
            }
            for m in all_metrics.values()
        ]
        summary_df = pd.DataFrame(rows)
        try:
            from IPython.display import display

            display(summary_df)
        except Exception:
            print(summary_df.to_string(index=False))

        # ── McNemar's test ───────────────────────────────────────────
        if "DORADO" in all_metrics and "SFT" in all_metrics:
            d = np.array(all_metrics["DORADO"]["correct_flags"])
            s = np.array(all_metrics["SFT"]["correct_flags"])
            b = int(np.sum(~d & s))
            c = int(np.sum(d & ~s))
            p_value = mcnemar_exact_pvalue(b, c)
            diff = all_metrics["DORADO"]["accuracy"] - all_metrics["SFT"]["accuracy"]
            print("\n--- STATISTICAL SIGNIFICANCE TEST ---")
            print(f"McNemar's Test (DORADO vs SFT):")
            print(f"  Improvement: {diff:.1%}")
            print(f"  Discordant pairs: b={b}, c={c}")
            print(f"  p-value: {p_value:.4f}")
            sig = "✅ Significant" if p_value < 0.05 else "⚠️ NOT significant"
            print(f"  {sig} at α=0.05")

    return all_metrics, all_results
