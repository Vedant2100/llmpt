# TODO / Design Decisions

## Reward Model Training Strategy

**Current Implementation:** Train fresh RM every round (adaptive approach)

### Design Choice
The current implementation trains a new reward model in each iterative DPO round:
- **Round 1:** Generate from SFT → Label → Train RM₁ → Re-label with RM₁ → DPO₁
- **Round 2:** Generate from DPO₁ → Label → Train RM₂ → Re-label with RM₂ → DPO₂
- **Round 3:** Generate from DPO₂ → Label → Train RM₃ → Re-label with RM₃ → DPO₃

### Rationale
- ✅ **Adaptive:** RM adapts to changing candidate quality/distribution each round
- ✅ **Robust:** Avoids potential distribution shift issues with fixed RM
- ❌ **Compute:** Re-trains RM every round (more expensive)

### Alternative Considered: One-Time RM Training
Train RM once before iterations, reuse for all rounds:
- ✅ **Efficient:** Train RM only once
- ❌ **Fixed:** RM trained on Round 1 candidates may not generalize well to improved Round 2+ candidates
- ❌ **Distribution shift:** As policy improves, candidate distribution changes

### Status
**Current:** Using per-round RM training
**Future consideration:** Could add config flag `train_rm_per_round: [True, False]` to test both approaches

---

## Other TODOs
- [ ] Consider adding per-round RM training as configurable parameter
- [ ] Benchmark compute cost difference between approaches
- [ ] Test if fixed RM generalizes to improved candidate distributions

---

## Paper Alignment Tracker

| Gap | Paper Intent | Previous Behavior | Implemented Change | Validation Metric | Status |
|---|---|---|---|---|---|
| Pair construction | Drop all-wrong questions; correctness-first pairing | All-wrong could still produce length-heuristic pairs | Removed heuristic fallback from training path; all-wrong now dropped | `all_wrong_dropped_questions > 0` and `length_heuristic_pairs == 0` | ✅ Done |
| Dual-reward gating | Hard correctness gate, RM mainly for correct-correct quality ranking | Correctness and RM were additively mixed across all candidates | Correctness-first gating with C>I primary pairs; RM adds C>C ranking when enabled | Positive C>I count; C>C pairs appear only with RM | ✅ Done |
| SFT↔DPO↔Eval consistency | One consistent policy/reference contract | Mixed merge vs stack behavior caused adapter-key mismatch risk | Declared and enforced merged-SFT DPO contract; eval loads adapters consistently | No missing adapter-key warnings during DORADO eval | ✅ Done |
| Verifier contract robustness | Rule parsing resilient to format variation | GT/parse heavily tied to strict `####` splitting and integer regex | Added canonical numeric parsing (ints/decimals/fractions, signed), unified GT extraction helper | Higher/steady parsed ratio with fewer formatting false negatives | ✅ Done |
| Evaluation objective breadth | Track correctness plus response quality signals | Primarily exact-match accuracy | Added marker compliance and answer diversity ratios to eval outputs/summary | Accuracy + Parsed + Marker + Diversity reported together | ✅ Done |

### Alternative Contract (Not Active)

- **Strict stacked-adapter semantics:** keep SFT as a separate adapter during DPO and evaluation.
- **Flow:** BASE → SFT adapter (frozen) → DORADO adapter for both training and eval.
- **Use case:** explicit measurement of "DORADO incremental gain over SFT" without merged-weight baking.
- **Caveat:** if enabled, train/eval must both use this path to avoid adapter-key mismatch artifacts.
