"""Stage 3: Preference-pair labeling (verifiable correctness ± learned RM)."""

import os
import random

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
    """Build preference pairs using correctness and optionally RM scoring.

    When no correct/incorrect distinction exists for a question, falls back
    to a length-based heuristic (longer chain-of-thought = chosen) so that
    DPO always has training signal.

    Returns ``(pairs, labels, pair_stats)``.
    """
    pairs: list[tuple[str, str, str]] = []
    labels: list[int] = []
    pair_stats = {
        "num_pairs": 0,
        "correct_incorrect_pairs": 0,
        "correct_correct_pairs": 0,
        "length_heuristic_pairs": 0,
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
        from dorado.config import make_bnb_config

        bnb_config = make_bnb_config(exp_config)
        load_kwargs = dict(num_labels=2, device_map="auto", torch_dtype=torch.float16)
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
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
        correctness_score = exp_config["correctness_weight"] if is_correct else 0.0

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

        return correctness_score + rm_score, is_correct, rm_score

    # ── build pairs ──────────────────────────────────────────────────
    all_wrong_count = 0
    no_pair_count = 0
    for q, samples in tqdm(all_samples.items(), desc="Labeling candidates"):
        scored = [(s, *score_response(q, s)) for s in samples]
        # scored: list of (response_text, total_score, is_correct, rm_score)
        scored.sort(key=lambda x: x[1], reverse=True)

        if not any(s[2] for s in scored):  # no candidate is correct
            all_wrong_count += 1

        n = len(scored)
        made_pair = False
        for i in range(n // 2):
            best, worst = i, n - 1 - i
            if scored[best][1] > scored[worst][1]:
                pairs.append((q, scored[best][0], scored[worst][0]))
                labels.append(1)
                pair_stats["num_pairs"] += 1
                made_pair = True
                if scored[best][2] and not scored[worst][2]:  # correct vs incorrect
                    pair_stats["correct_incorrect_pairs"] += 1
                elif scored[best][2] and scored[worst][2]:  # both correct
                    pair_stats["correct_correct_pairs"] += 1

        # Fallback: if all candidates scored the same (e.g. all wrong),
        # prefer the longer response (proxy for more reasoning steps).
        if not made_pair and n >= 2:
            by_len = sorted(scored, key=lambda x: len(x[0]), reverse=True)
            if len(by_len[0][0]) > len(by_len[-1][0]):
                pairs.append((q, by_len[0][0], by_len[-1][0]))
                labels.append(1)
                pair_stats["num_pairs"] += 1
                pair_stats["length_heuristic_pairs"] += 1
                made_pair = True

        if not made_pair:
            no_pair_count += 1

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
    if all_wrong_count > 0:
        pipeline_warn(
            f"Labeling: {all_wrong_count}/{total_q} questions had zero correct "
            f"candidates (all wrong)."
        )
    if no_pair_count > 0:
        pipeline_warn(
            f"Labeling: {no_pair_count}/{total_q} questions produced zero "
            f"preference pairs (even after length-heuristic fallback)."
        )
    if pair_stats["correct_incorrect_pairs"] == 0 and pair_stats["num_pairs"] > 0:
        pipeline_warn(
            "Labeling: zero correct-vs-incorrect pairs. All pairs are from "
            "length-heuristic fallback or correct-vs-correct comparisons."
        )
    heuristic_ratio = pair_stats["length_heuristic_pairs"] / max(
        pair_stats["num_pairs"], 1
    )
    if heuristic_ratio > 0.5 and pair_stats["num_pairs"] > 0:
        pipeline_warn(
            f"Labeling: {heuristic_ratio:.0%} of pairs are from length-heuristic "
            f"fallback. Training signal may be noisy."
        )
    if pair_stats["num_pairs"] == 0:
        pipeline_warn("Labeling: produced zero preference pairs total.")

    print(f"✅ Created {pair_stats['num_pairs']} preference pairs")
    print(f"   - Correct vs Incorrect: {pair_stats['correct_incorrect_pairs']}")
    print(f"   - Correct vs Correct: {pair_stats['correct_correct_pairs']}")
    print(f"   - Length heuristic:   {pair_stats['length_heuristic_pairs']}")
    if pair_stats["rm_scores_used"]:
        print(f"   - Avg RM Score: {pair_stats['avg_rm_score']:.3f}")

    return pairs, labels, pair_stats
