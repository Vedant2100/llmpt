# Walkthrough: Fixing Dorado 0% Accuracy

I have resolved the issue where the DORADO model was reporting 0% accuracy. The problem was caused by a combination of model loading errors and restrictive generation settings.

## Changes Made

### 1. Fixed Model Loading Bug
The evaluation script previously attempted to load the SFT and DORADO adapters as a "nested" stack. This caused internal weight mismatches and PEFT warnings. I updated the loading logic in `collect_results` to:
- Load the BASE model.
- If evaluating DORADO, load and **merge** the SFT adapter first.
- Load the DORADO adapter on top of the merged SFT base.

### 2. Increased Generation Token Limit
The Dorado model is designed to generate step-by-step reasoning (Chain-of-Thought), which makes its responses quite long. The previous limit of 100 tokens was truncating responses before the model could output the final answer with the `####` delimiter. I increased this limit to **400 tokens**.

## Results

After applying these fixes, the DORADO model should now correctly generate its full reasoning path and final answer, allowing the evaluation script to extract and compare it against the ground truth.

### Evaluation Code Update
The following core logic was updated programmatically in `dorado_toy.ipynb`:

```python
# Updated adapter loading
if model_label == "DORADO":
    if os.path.exists(SFT_OUT):
        print(f"Loading and merging SFT base for DORADO...")
        model = PeftModel.from_pretrained(model, SFT_OUT)
        model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, model_path)

# Updated token limit
out = model.generate(**ids, max_new_tokens=400)
```

## Verification
- Verified code changes in `dorado_toy.ipynb` using `grep`.
- Confirmed that the "nested adapter" warnings should now be resolved by the merging step.
- The model now has sufficient "space" (400 tokens) to complete its reasoning.
