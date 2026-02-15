# Suggestions to Improve Dorado Experiment

The current notebook is a strictly "toy" implementation designed for speed and low hardware requirements. To transition to a meaningful reasoning model, consider the following improvements:

## 🚀 1. Data Scale-Up (Critical)
- **Problem**: The experiment uses 50 SFT samples and 5 DPO samples. Models cannot "learn" complex mathematical reasoning from this few examples.
- **Improved**: 
    - **SFT**: Use 2,000+ samples from datasets like *MathInstruct* or *Open-Web-Math*.
    - **DPO**: Generate 1,000+ high-quality preference pairs. Performance starts to plateau around 5k-10k pairs for small models.

## 🧠 2. Increase Candidate Diversity (n)
- **Problem**: `n=2` candidates per question limits the model's exposure to different reasoning paths.
- **Improved**: Use `n=4` or `n=8`. The paper uses `n=5`. More samples allow for more "hard negatives" (responses that look correct but are wrong) and "high-quality positives" (responses with clear Chain-of-Thought).

## 🏆 3. Reward Model (RM) Enhancement
- **Problem**: A 0.5B model is a weak scorer. 
- **Improved**: 
    - Use a larger base for the RM (e.g., Qwen2-1.5B or 7B) even if the policy model is small.
    - **Training Data**: Instead of just using generated samples, include a separate "Golden Preference" dataset like *HelpSteer* or *Orca* to ground the RM in better language quality.

## 🔄 4. Iterative Dorado Cycles
- **Problem**: The current pipeline is a single "offline" pass.
- **Improved**: Implement **Iterative DPO**.
    1. Train Stage 1.
    2. Generate candidates and train DPO.
    3. Use the DPO model as the *new* generator for the next round of labeling.
    4. Repeat. This is how models like *DeepSeek-R1* achieve state-of-the-art results.

## 🔍 5. Robust Verification
- **Problem**: The regex extraction is simple and might miss non-standard formatting.
- **Improved**: 
    - Use a **Python Executor** to verify results if the math problems involve code-like steps.
    - Incorporate **LLM-as-a-Judge** for Stage 3 labeling to verify not just the *answer* but the *process*.

## ⚙️ 6. Hyperparameter Tuning
- **LoRA r/alpha**: Increase `r` to 32 or 64 to allow the model to capture more complex reasoning weights.
- **DPO Beta**: Adjust the `beta` parameter (DPO's divergence penalty). A higher beta makes the model stay closer to the SFT base; a lower beta allows more exploration.
- **Gradient Accumulation**: Increase effective batch size via gradient accumulation (e.g., 32 or 64) for more stable convergence.

## 📝 7. Quality over Quantity in SFT 
- **Problem**: Alpaca is general chat. 
- **Improved**: Switch to **Chain-of-Thought (CoT)** SFT data. If the model starts with a "thinking" habit, DPO will be much more effective at refining that thinking.
