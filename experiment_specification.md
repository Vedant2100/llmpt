# Dorado Toy Experiment Specification

This document outlines the current configuration and technical specifications for the Dorado toy implementation in `dorado_toy.ipynb`.

## 🏗️ Model Architecture
- **Base Model**: `Qwen/Qwen2.5-0.5B`
- **Architecture**: Causal Language Model (Transformer-based)
- **Quantization**: 4-bit (NF4) used across all training stages to reduce VRAM usage.

## 📈 Pipeline Stages

### Stage 0: Environment Setup
- **Dependencies**: `transformers`, `datasets`, `trl`, `peft`, `bitsandbytes`, `accelerate`.
- **Hardware Profile**: Optimized for NVIDIA T4 (GCP/Colab standard).

### Stage 1: Cold-Start SFT
- **Dataset**: `tatsu-lab/alpaca` (50 samples).
- **LoRA Config**: `r=8`, `alpha=16`, Target: `q_proj`, `v_proj`.
- **Training**: 1 Epoch, Learning Rate (default), Batch Size 4.
- **Objective**: Improve general follow-instruction capabilities on non-verifiable data.

### Stage 2: Candidate Generation
- **Dataset**: `openai/gsm8k` (5 samples).
- **Generation Params**: `n=2` sequences per prompt, `temperature=0.7`, `max_new_tokens=150`.
- **Source Model**: Stage 1 SFT adapter loaded over Base.

### Stage 3: Labeling (Dual Rewards)
- **Verifiable Reward**: Exact numeric match against GSM8K ground truth.
- **Preference Logic**:
    - **Positive (Chosen)**: Correct response.
    - **Negative (Rejected)**: Incorrect response.
    - **Tie-breaking**: Preference between multiple correct responses (sequential).

### Stage 4: Reward Model Training
- **Model**: `Qwen2ForSequenceClassification` (0.5B).
- **LoRA Config**: `r=16`, `alpha=32`, Target: `q_proj`, `v_proj`.
- **Training**: 2 Epochs, Batch Size 4, Max Length 512.
- **Objective**: Learn to score quality among "correct" responses.

### Stage 5: Offline DPO Training
- **Adapter Base**: BASE model merged with Stage 1 SFT adapter.
- **LoRA Config**: `r=16`, `alpha=32`, Target: `q_proj`, `v_proj`.
- **Training**: 1 Epoch, Batch Size 2.
- **Objective**: Align model to prefer high-quality reasoning using both verifiable and learned rewards.

### Stage 6: Evaluation
- **Models Compared**: BASE vs. SFT vs. DORADO.
- **Evaluation Set**: First 5 questions of GSM8K.
- **Generation Params**: `max_new_tokens=400` (CoT-friendly).
- **Metric**: Exact Match Accuracy (extracted via regex).
