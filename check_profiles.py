"""Quick sanity check — run this in the pod: python check_profiles.py"""
from dorado.config import PROFILES, get_profile

assert set(PROFILES.keys()) == {"fast", "full"}, f"Got: {set(PROFILES.keys())}"
f = get_profile("fast")
u = get_profile("full")

checks = {
    "both full fine-tune":         f["finetuning_type"] == u["finetuning_type"] == "full",
    "both grad_accum=8":           f["gradient_accumulation_steps"] == u["gradient_accumulation_steps"] == 8,
    "both armo RM":                f["rm_strategy"] == u["rm_strategy"] == "armo",
    "both vllm eval":              f["eval_engine"] == u["eval_engine"] == "vllm",
    "both deepspeed Z3":           f["deepspeed_config"] == u["deepspeed_config"] == "examples/deepspeed/ds_z3_config.json",
    "both dpo_beta=0.1":           f["dpo_beta"] == u["dpo_beta"] == 0.1,
    "both dpo_lr=5e-7":            f["dpo_lr"] == u["dpo_lr"] == 5e-7,
    "full sft_samples=5000":       u["sft_samples"] == 5000,
    "fast sft_samples=200":        f["sft_samples"] == 200,
    "fast model=1.5B":             "1.5B" in f["base_model"],
    "full model=7B":               "7B" in u["base_model"],
}

all_ok = True
for name, ok in checks.items():
    status = "✅" if ok else "❌"
    print(f"  {status} {name}")
    if not ok:
        all_ok = False

if all_ok:
    print("\n✅ All profile checks passed")
else:
    raise SystemExit("❌ Some checks failed")
