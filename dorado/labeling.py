"""Stage 3: Preference-pair labeling (verifiable correctness ± learned RM)."""

import os

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

from dorado.utils import clear_gpu, extract_answer_from_response, pipeline_warn


def run_labeling_stage(
    exp_config: dict,
    all_samples: dict,
    gt: dict,
    use_rm: bool = False,
) -> tuple[list, list, dict]:
    """Build preference pairs with paper-style dual reward gating.

    Policy:
    - First gate by verifiable correctness.
    - Build correct-vs-incorrect pairs when possible.
    - If all candidates are correct and RM is enabled, build correct-vs-correct
      quality pairs using RM max/min ranking.
    - Drop questions with all-wrong candidates (no heuristic fallback).

    Returns ``(pairs, labels, pair_stats)``.
    """
    pairs: list[tuple[str, str, str]] = []
    labels: list[int] = []
    pair_stats = {
        "num_pairs": 0,
        "correct_incorrect_pairs": 0,
        "correct_correct_pairs": 0,
        "length_heuristic_pairs": 0,
        "all_wrong_dropped_questions": 0,
        "all_correct_questions": 0,
        "mixed_questions": 0,
        "avg_rm_score": 0.0,
        "rm_scores_used": [],
    }

    # ── optionally load RM ───────────────────────────────────────────
    rm_model = None
    rm_tokenizer = None
    if use_rm and not os.path.exists("reward_model"):
        pipeline_warn(
            "Labeling: use_rm=True but 'reward_model/' not found. "
            "Falling back to correctness-only scoring."
        )
    if use_rm and os.path.exists("reward_model"):
        print("Loading reward model for scoring...")
        BASE = exp_config["rm_base_model"]
        from dorado.config import make_model_load_kwargs

        load_kwargs = make_model_load_kwargs(exp_config, num_labels=2)
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            BASE, **load_kwargs
        )
        rm_model = PeftModel.from_pretrained(rm_model, "reward_model")
        rm_tokenizer = AutoTokenizer.from_pretrained(BASE)
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
        rm_model.eval()

    # ── scoring helper ───────────────────────────────────────────────
    def score_response(question: str, response: str):
        gt_answer = gt[question]
        predicted = extract_answer_from_response(response)
        is_correct = predicted == gt_answer

        rm_score = 0.0
        if rm_model is not None:
            text = question + " [ANS] " + response
            inputs = rm_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            ).to(rm_model.device)
            with torch.no_grad():
                logits = rm_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                rm_score = probs[0][1].item() * exp_config["rm_weight"]
            pair_stats["rm_scores_used"].append(probs[0][1].item())

        return is_correct, rm_score

    # ── build pairs ──────────────────────────────────────────────────
    all_wrong_count = 0
    all_correct_count = 0
    mixed_count = 0
    no_pair_count = 0
    for q, samples in tqdm(all_samples.items(), desc="Labeling candidates"):
        scored = []
        for response in samples:
            is_correct, rm_score = score_response(q, response)
            scored.append(
                {
                    "response": response,
                    "is_correct": is_correct,
                    "rm_score": rm_score,
                    "length": len(response),
                }
            )

        correct = [row for row in scored if row["is_correct"]]
        incorrect = [row for row in scored if not row["is_correct"]]

        if not correct:
            all_wrong_count += 1
            no_pair_count += 1
            continue

        if not incorrect:
            all_correct_count += 1
            pair_stats["all_correct_questions"] += 1
            if use_rm and len(correct) >= 2:
                ranked = sorted(correct, key=lambda x: x["rm_score"], reverse=True)
                if ranked[0]["rm_score"] > ranked[-1]["rm_score"]:
                    pairs.append((q, ranked[0]["response"], ranked[-1]["response"]))
                    labels.append(1)
                    pair_stats["num_pairs"] += 1
                    pair_stats["correct_correct_pairs"] += 1
                else:
                    no_pair_count += 1
            else:
                no_pair_count += 1
            continue

        mixed_count += 1
        pair_stats["mixed_questions"] += 1
        chosen = max(correct, key=lambda x: x["length"])
        rejected = min(incorrect, key=lambda x: x["length"])

        pairs.append((q, chosen["response"], rejected["response"]))
        labels.append(1)
        pair_stats["num_pairs"] += 1
        pair_stats["correct_incorrect_pairs"] += 1

        if use_rm and len(correct) >= 2:
            ranked = sorted(correct, key=lambda x: x["rm_score"], reverse=True)
            if ranked[0]["rm_score"] > ranked[-1]["rm_score"]:
                pairs.append((q, ranked[0]["response"], ranked[-1]["response"]))
                labels.append(1)
                pair_stats["num_pairs"] += 1
                pair_stats["correct_correct_pairs"] += 1

    # ── cleanup ──────────────────────────────────────────────────────
    if rm_model is not None:
        del rm_model, rm_tokenizer
        clear_gpu()

    if pair_stats["rm_scores_used"]:
        pair_stats["avg_rm_score"] = sum(pair_stats["rm_scores_used"]) / len(
            pair_stats["rm_scores_used"]
        )

    # ── diagnostic warnings ──────────────────────────────────────────
    total_q = len(all_samples)
    pair_stats["all_wrong_dropped_questions"] = all_wrong_count
    if all_wrong_count > 0:
        pipeline_warn(
            f"Labeling: {all_wrong_count}/{total_q} questions had zero correct "
            f"candidates (all wrong) and were dropped."
        )
    if no_pair_count > 0:
        pipeline_warn(
            f"Labeling: {no_pair_count}/{total_q} questions produced zero "
            f"preference pairs after correctness-first gating."
        )
    if pair_stats["correct_incorrect_pairs"] == 0 and pair_stats["num_pairs"] > 0:
        pipeline_warn(
            "Labeling: zero correct-vs-incorrect pairs. All pairs are from "
            "correct-vs-correct comparisons."
        )
    if pair_stats["num_pairs"] == 0:
        pipeline_warn("Labeling: produced zero preference pairs total.")

    print(f"✅ Created {pair_stats['num_pairs']} preference pairs")
    print(f"   - Correct vs Incorrect: {pair_stats['correct_incorrect_pairs']}")
    print(f"   - Correct vs Correct: {pair_stats['correct_correct_pairs']}")
    print(f"   - Length heuristic:   {pair_stats['length_heuristic_pairs']}")
    print(f"   - All-wrong dropped:  {all_wrong_count}")
    print(f"   - All-correct qs:     {all_correct_count}")
    print(f"   - Mixed qs:           {mixed_count}")
    if pair_stats["rm_scores_used"]:
        print(f"   - Avg RM Score: {pair_stats['avg_rm_score']:.3f}")

    return pairs, labels, pair_stats
