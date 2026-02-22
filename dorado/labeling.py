"""Stage 3: Preference-pair labeling (verifiable correctness ± learned RM)."""

import os

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

from dorado.utils import clear_gpu, extract_answer_from_response


def run_labeling_stage(
    exp_config: dict,
    all_samples: dict,
    gt: dict,
    use_rm: bool = False,
) -> tuple[list, list, dict]:
    """Build preference pairs using correctness and optionally RM scoring.

    Returns ``(pairs, labels, pair_stats)``.
    """
    pairs: list[tuple[str, str, str]] = []
    labels: list[int] = []
    pair_stats = {
        "num_pairs": 0,
        "correct_incorrect_pairs": 0,
        "correct_correct_pairs": 0,
        "avg_rm_score": 0.0,
        "rm_scores_used": [],
    }

    # ── optionally load RM ───────────────────────────────────────────
    rm_model = None
    rm_tokenizer = None
    if use_rm and os.path.exists("reward_model"):
        print("Loading reward model for scoring...")
        BASE = exp_config["rm_base_model"]
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            BASE, num_labels=2, quantization_config=bnb_config, device_map="auto"
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
    for q, samples in tqdm(all_samples.items(), desc="Labeling candidates"):
        scored = [(s, *score_response(q, s)) for s in samples]
        scored.sort(key=lambda x: x[1], reverse=True)

        n = len(scored)
        for i in range(n // 2):
            best, worst = i, n - 1 - i
            if scored[best][1] > scored[worst][1]:
                pairs.append((q, scored[best][0], scored[worst][0]))
                labels.append(1)
                pair_stats["num_pairs"] += 1
                if scored[best][2] and not scored[worst][2]:
                    pair_stats["correct_incorrect_pairs"] += 1
                elif scored[best][2] and scored[worst][2]:
                    pair_stats["correct_correct_pairs"] += 1

    # ── cleanup ──────────────────────────────────────────────────────
    if rm_model is not None:
        del rm_model, rm_tokenizer
        clear_gpu()

    if pair_stats["rm_scores_used"]:
        pair_stats["avg_rm_score"] = sum(pair_stats["rm_scores_used"]) / len(
            pair_stats["rm_scores_used"]
        )

    print(f"✅ Created {pair_stats['num_pairs']} preference pairs")
    print(f"   - Correct vs Incorrect: {pair_stats['correct_incorrect_pairs']}")
    print(f"   - Correct vs Correct: {pair_stats['correct_correct_pairs']}")
    if pair_stats["rm_scores_used"]:
        print(f"   - Avg RM Score: {pair_stats['avg_rm_score']:.3f}")

    return pairs, labels, pair_stats
