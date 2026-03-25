"""Stage 3: Preference-pair labeling with ArmoRM (dual reward)."""

import os

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dorado.utils import clear_gpu, pipeline_warn


class ArmoRMScorer:
    """Score responses using pre-trained ArmoRM (non-verifiable reward)."""

    def __init__(
        self,
        model_id: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
        max_length: int = 4096,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.max_length = max_length
        self.device = self.model.device

    def score_batch(self, message_pairs: list[list[dict]]) -> list[float]:
        """Score a batch of conversation message lists.

        Each element is a list of messages, e.g.:
        [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        """
        input_ids = self.tokenizer.apply_chat_template(
            message_pairs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            scores = outputs.logits.squeeze(-1).float().tolist()
        # Handle single-item batch
        if isinstance(scores, float):
            scores = [scores]
        return scores

    def score_single(self, question: str, response: str) -> float:
        """Score a single question-response pair."""
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]
        return self.score_batch([messages])[0]

    def cleanup(self):
        del self.model
        del self.tokenizer
        clear_gpu()


def _load_math_answer_checker():
    """Load the eval/ answer checking utilities for verifiable reward."""
    import sys
    # Add reference/eval/ to path so we can import its utils
    eval_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference", "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    try:
        from utils.parser import extract_answer
        from utils.grader import check_is_correct
        return extract_answer, check_is_correct
    except ImportError:
        pipeline_warn(
            "Could not import eval/utils for \\boxed{} answer checking. "
            "Falling back to simple #### extraction."
        )
        return None, None


def run_labeling_stage(
    exp_config: dict,
    all_samples: dict,
    gt: dict,
) -> tuple[list, list, dict]:
    """Build preference pairs with paper-style dual reward gating.

    Policy (matching Dorado paper exactly):
    1. Score all candidates for correctness (verifiable reward).
    2. For questions with mixed correct/incorrect: pair correct vs incorrect.
    3. For questions with all correct: use ArmoRM to rank by quality,
       pair highest-scored vs lowest-scored (non-verifiable reward).
    4. Drop questions with all incorrect candidates.

    Returns ``(pairs, labels, pair_stats)``.
    """
    rm_strategy = exp_config.get("rm_strategy", "armo")

    pairs: list[tuple[str, str, str]] = []
    labels: list[int] = []
    pair_stats = {
        "num_pairs": 0,
        "correct_incorrect_pairs": 0,
        "correct_correct_pairs": 0,
        "total_candidates": 0,
        "format_compliant_candidates": 0,
        "all_wrong_dropped_questions": 0,
        "all_correct_questions": 0,
        "mixed_questions": 0,
        "avg_rm_score": 0.0,
        "rm_scores_used": [],
    }

    # ── load answer checker ──────────────────────────────────────────
    extract_answer, check_is_correct = _load_math_answer_checker()
    use_boxed = extract_answer is not None

    # ── optionally load ArmoRM ───────────────────────────────────────
    rm_scorer = None
    if rm_strategy == "armo":
        print("Loading ArmoRM for non-verifiable reward scoring...")
        rm_scorer = ArmoRMScorer()
        print("✅ ArmoRM loaded")

    # ── scoring helper ───────────────────────────────────────────────
    def check_correctness(question: str, response: str, gt_answer: str) -> bool:
        """Check if response contains the correct answer."""
        if use_boxed:
            predicted = extract_answer(response, "math")
            return check_is_correct(predicted, gt_answer)
        else:
            # Fallback: simple #### extraction
            from dorado.utils import extract_answer_from_response
            predicted = extract_answer_from_response(response)
            return predicted == gt_answer

    # ── build pairs ──────────────────────────────────────────────────
    for q, samples in tqdm(all_samples.items(), desc="Labeling candidates"):
        gt_answer = gt.get(q, "")
        scored = []

        for response in samples:
            is_correct = check_correctness(q, response, gt_answer)
            pair_stats["total_candidates"] += 1

            # Check format compliance
            if "\\boxed" in response or "####" in response:
                pair_stats["format_compliant_candidates"] += 1

            scored.append({
                "response": response,
                "is_correct": is_correct,
                "rm_score": 0.0,
            })

        correct = [s for s in scored if s["is_correct"]]
        incorrect = [s for s in scored if not s["is_correct"]]

        # Case 1: All wrong → drop
        if not correct:
            pair_stats["all_wrong_dropped_questions"] += 1
            continue

        # Case 2: Mixed correct/incorrect → pair correct vs incorrect
        if incorrect:
            pair_stats["mixed_questions"] += 1
            # Pick best correct and worst incorrect
            chosen = correct[0]
            rejected = incorrect[0]
            pairs.append((q, chosen["response"], rejected["response"]))
            labels.append(1)
            pair_stats["num_pairs"] += 1
            pair_stats["correct_incorrect_pairs"] += 1

            # If multiple correct/incorrect, add a second pair
            if len(correct) >= 2 and len(incorrect) >= 2:
                pairs.append((q, correct[1]["response"], incorrect[1]["response"]))
                labels.append(1)
                pair_stats["num_pairs"] += 1
                pair_stats["correct_incorrect_pairs"] += 1
            continue

        # Case 3: All correct → use ArmoRM to rank by quality
        pair_stats["all_correct_questions"] += 1
        if rm_scorer is not None and len(correct) >= 2:
            # Score all correct responses with ArmoRM
            for s in correct:
                s["rm_score"] = rm_scorer.score_single(q, s["response"])
                pair_stats["rm_scores_used"].append(s["rm_score"])

            ranked = sorted(correct, key=lambda x: x["rm_score"], reverse=True)
            if ranked[0]["rm_score"] > ranked[-1]["rm_score"]:
                pairs.append((q, ranked[0]["response"], ranked[-1]["response"]))
                labels.append(1)
                pair_stats["num_pairs"] += 1
                pair_stats["correct_correct_pairs"] += 1

    # ── cleanup ──────────────────────────────────────────────────────
    if rm_scorer is not None:
        rm_scorer.cleanup()

    if pair_stats["rm_scores_used"]:
        pair_stats["avg_rm_score"] = (
            sum(pair_stats["rm_scores_used"]) / len(pair_stats["rm_scores_used"])
        )

    # ── diagnostics ──────────────────────────────────────────────────
    total_q = len(all_samples)
    if pair_stats["all_wrong_dropped_questions"] > 0:
        pipeline_warn(
            f"Labeling: {pair_stats['all_wrong_dropped_questions']}/{total_q} questions "
            f"had zero correct candidates and were dropped."
        )
    if pair_stats["num_pairs"] == 0:
        pipeline_warn("Labeling: produced zero preference pairs total.")

    if pair_stats["total_candidates"] > 0:
        format_ratio = (
            pair_stats["format_compliant_candidates"] / pair_stats["total_candidates"]
        )
        if format_ratio < 0.5:
            pipeline_warn(
                f"Labeling: only {format_ratio:.0%} candidates have answer markers."
            )

    print(f"✅ Created {pair_stats['num_pairs']} preference pairs")
    print(f"   - Correct vs Incorrect: {pair_stats['correct_incorrect_pairs']}")
    print(f"   - Correct vs Correct (ArmoRM): {pair_stats['correct_correct_pairs']}")
    print(f"   - All-wrong dropped: {pair_stats['all_wrong_dropped_questions']}")
    print(f"   - All-correct questions: {pair_stats['all_correct_questions']}")
    print(f"   - Mixed questions: {pair_stats['mixed_questions']}")
    if pair_stats["rm_scores_used"]:
        print(f"   - Avg ArmoRM Score: {pair_stats['avg_rm_score']:.3f}")

    return pairs, labels, pair_stats
