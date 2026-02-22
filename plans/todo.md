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
