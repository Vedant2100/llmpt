# Plan: Parameter Sweep Infrastructure with Iterative DPO & Dual Preferences

This plan transforms [dorado_toy.ipynb](dorado_toy.ipynb) from a single-config toy implementation into a full experiment harness supporting parameter sweeps, dual-preference DPO (verifiable + learned RM scoring), iterative training rounds, automatic result export to Excel, and artifact cleanup.

**Context**: Currently 82% of config parameters are unused (17 defined, only 3 used). Training stages have hardcoded values scattered across 6 cells. The reward model is trained but never applied to score candidates. No iterative DPO loop exists despite having an `ITERATIVE_DPO_ROUNDS` parameter. Results are displayed but never saved.

**Steps**

1. **Restructure [dorado_toy.ipynb](dorado_toy.ipynb) Cell 1: Parameter Grid Configuration**
   - Replace scalar parameters with lists grouped by meta-categories:
     - `DATASET_CONFIG`: `sft_samples`, `dpo_pairs`, `candidates_per_question`, `sft_dataset_name`, `dpo_dataset_name`, `eval_split`, `eval_max_samples`
     - `MODEL_CONFIG`: `base_model`, `rm_base_model`
     - `ARCHITECTURE_CONFIG`: `lora_r`, `lora_alpha`, `dpo_beta`, `gradient_accumulation_steps`
     - `TRAINING_CONFIG`: `iterative_dpo_rounds`, `sft_epochs`, `rm_epochs`, `dpo_epochs`, `sft_batch_size`, `rm_batch_size`, `dpo_batch_size`
     - `GENERATION_CONFIG`: `temperature`, `max_new_tokens_gen`, `max_new_tokens_eval`
     - `EVAL_CONFIG`: `eval_batch_size`
     - `DUAL_PREFERENCE_CONFIG`: `use_rm_scoring`, `rm_weight`, `correctness_weight`
   - Add `build_experiment_grid()` function using `itertools.product()` to generate all parameter combinations
   - Add `estimate_time(exp_config)` function computing per-experiment runtime (SFT + generation + RM + DPO × rounds + eval)
   - Print experiment grid summary with first 3 configs and total time estimate
   - Initialize `RESULTS_LOG = []` and timestamped `RESULTS_FILE` path

2. **Refactor Cell 9 (SFT Stage): Parameterize Training**
   - Replace hardcoded `50` samples with `exp_config['sft_samples']`
   - Replace hardcoded `"tatsu-lab/alpaca"` with `exp_config['sft_dataset_name']`
   - Replace LoRA `r=8, alpha=16` with `exp_config['lora_r'], exp_config['lora_alpha']`
   - Replace `per_device_train_batch_size=4` with `exp_config['sft_batch_size']`
   - Replace `num_train_epochs=1` with `exp_config['sft_epochs']`
   - Add `gradient_accumulation_steps` from config
   - Keep output directory as `coldstart_dorado/` (will be cleaned after experiment)

3. **Refactor Cell 11 (Candidate Generation): Parameterize Sampling**
   - Replace hardcoded `5` GSM8K samples with `exp_config['dpo_pairs']`
   - Replace `num_return_sequences=2` with `exp_config['candidates_per_question']`
   - Replace `temperature=0.7` with `exp_config['temperature']`
   - Replace `max_new_tokens=150` with `exp_config['max_new_tokens_gen']`
   - Keep `ALL_SAMPLES`, `GT`, `QUESTIONS` as outputs

4. **Implement Dual Preference Scoring in Cell 13 (Labeling)**
   - Keep existing verifiable correctness logic (regex extraction + ground truth match)
   - **New**: Add conditional RM scoring when `exp_config['use_rm_scoring'] == True`:
     - Load trained RM from `reward_model/` directory
     - For each candidate, compute RM score: `score = RM(question + " [ANS] " + response)`
     - Combine scores: `final_score = correctness_weight * is_correct + rm_weight * rm_score`
   - Create preference pairs using combined scores:
     - Sort candidates by `final_score`
     - Create pairs: highest vs lowest, 2nd highest vs 2nd lowest, etc.
   - Add `pair_stats` dict tracking: `{num_pairs, correct_correct_pairs, correct_incorrect_pairs, avg_rm_score}`
   - Output: `pairs`, `labels`, `pair_stats`

5. **Refactor Cell 15 (Reward Model Training): Parameterize + Support RM Base Selection**
   - Replace hardcoded `BASE` with `exp_config['rm_base_model']`
   - Replace LoRA `r=16, alpha=32` with `exp_config['lora_r'], exp_config['lora_alpha']`
   - Replace `per_device_train_batch_size=4` with `exp_config['rm_batch_size']`
   - Replace `num_train_epochs=2` with `exp_config['rm_epochs']`
   - Keep tokenization `max_length=512` (critical OOM fix)
   - Output: `reward_model/` directory

6. **Refactor Cell 18 (DPO Training): Parameterize Beta + Batch Size**
   - Replace LoRA `r=16, alpha=32` with `exp_config['lora_r'], exp_config['lora_alpha']`
   - Replace `per_device_train_batch_size=2` with `exp_config['dpo_batch_size']`
   - Replace `num_train_epochs=1` with `exp_config['dpo_epochs']`
   - Add `beta=exp_config['dpo_beta']` to `DPOConfig`
   - Add `gradient_accumulation_steps` from config
   - Keep `dorado_final/` output directory

7. **Implement Iterative DPO Loop (New Cell after Labeling)**
   - Wrap Cells 13-18 (Labeling → RM → DPO) in a `for round in range(exp_config['iterative_dpo_rounds']):` loop
   - **Round 1**: Use SFT model for generation (current behavior)
   - **Round 2+**: Use previous round's DPO model (`dorado_final/`) for generation
   - After each round:
     - Save round-specific artifacts: `dorado_round_{round}/`, `reward_model_round_{round}/`
     - Generate new candidates from updated model
     - Retrain RM on new preference pairs
     - Run DPO on new pairs
   - Track per-round metrics: `{round, num_pairs, avg_rm_score, pair_correctness_ratio}`

8. **Refactor Cell 20 (Evaluation): Parameterize + Capture Metrics**
   - Use `exp_config['eval_split']`, `exp_config['eval_max_samples']`, `exp_config['eval_batch_size']`
   - Replace `max_new_tokens=400` with `exp_config['max_new_tokens_eval']`
   - Evaluate: BASE, SFT (coldstart_dorado), DORADO final (dorado_final)
   - **New**: If iterative DPO used, evaluate all intermediate rounds: `dorado_round_1`, `dorado_round_2`, etc.
   - Capture comprehensive metrics per model:
     - `accuracy` (exact match %)
     - `avg_response_length` (tokens)
     - `num_parsed_answers` (how many had extractable numbers)
   - Store results in `experiment_results` dict with all params + metrics

9. **Create Experiment Orchestration Cell (New Cell 2)**
   - Add main experiment loop:
     ```python
     for exp_idx, exp_config in enumerate(EXPERIMENTS):
         print(f"\n{'='*60}\nExperiment {exp_idx+1}/{len(EXPERIMENTS)}\n{'='*60}")
         print(f"Config: {exp_config}")
         start_time = time.time()
         try:
             # Execute stages with exp_config
             # [Cells 9, 11, 13, 15, 18, 20 logic integrated]
             results = run_experiment(exp_config)
             results['status'] = 'success'
             results['runtime_minutes'] = (time.time() - start_time) / 60
         except Exception as e:
             results = {'status': 'failed', 'error': str(e), 'experiment_id': exp_config['experiment_id']}
         RESULTS_LOG.append({**exp_config, **results})
         cleanup_artifacts()  # Delete adapters after recording results
     ```
   - Add `cleanup_artifacts()` function: removes `coldstart_dorado/`, `reward_model*/`, `dorado_*/` directories
   - Add safety check: verify adequate disk space before each experiment

10. **Implement Excel Export (New Cell at End)**
    - Convert `RESULTS_LOG` to pandas DataFrame
    - Column organization:
      - Experiment metadata: `experiment_id`, `status`, `runtime_minutes`
      - All parameter columns (from config dicts)
      - Metrics: `base_accuracy`, `sft_accuracy`, `dorado_accuracy`, `improvement`
      - Pair stats: `num_pairs`, `correct_incorrect_ratio`, `avg_rm_score`
      - Per-round metrics (if iterative): `round_1_accuracy`, `round_2_accuracy`, etc.
    - Export using `df.to_excel(RESULTS_FILE, index=False, engine='openpyxl')`
    - Add conditional formatting: highlight best accuracy per parameter group
    - Print summary: total experiments, success rate, best config, accuracy distribution

11. **Add Progress Tracking & Checkpointing (New Utility Cell)**
    - Auto-save partial results after every 5 experiments: `RESULTS_FILE.replace('.xlsx', '_checkpoint.xlsx')`
    - Add resume capability: check for existing checkpoint, load `RESULTS_LOG`, skip completed experiment IDs
    - Log GPU memory peak after each stage using `torch.cuda.max_memory_allocated()`
    - Add experiment timer showing estimated time remaining based on completed runs

12. **Create Documentation Cell (New Markdown)**
    - Explain sweep usage:
      - How to modify parameter lists
      - How to interpret Excel output
      - How dual preference scoring works (verifiable + RM weights)
      - How iterative DPO improves over rounds
      - Storage requirements per experiment (~150MB, auto-cleaned)
    - Add example configs for common sweep scenarios:
      - Scale test: vary `sft_samples`, `dpo_pairs`
      - Architecture search: vary `lora_r`, `dpo_beta`
      - Dual preference comparison: `use_rm_scoring=[True, False]`
      - Iterative rounds: `iterative_dpo_rounds=[1, 2, 3]`

**Verification**
- Run minimal sweep (2 configs): `sft_samples=[50, 100]`, all other params single-value
- Check Excel output has correct columns: all params + accuracies
- Verify artifacts deleted after each experiment (disk usage stable)
- For iterative DPO config, confirm per-round accuracy tracking
- For dual preference config with `use_rm_scoring=True`, verify RM scores appear in pair_stats

**Decisions**
- **Full refactor over wrapper**: All stages now read from `exp_config` instead of hardcoded values, eliminating the 82% parameter disconnect
- **Dual preference implementation**: RM scores combined with verifiable correctness using weighted sum; supports pure-verifiable (weight=1.0, 0.0) or hybrid approaches
- **Automatic cleanup**: Adapters deleted after results captured to prevent storage exhaustion (user approved)
- **Iterative DPO**: Full implementation with per-round tracking, using previous round's model for next generation cycle
- **Per-round RM retraining**: Each iteration trains a fresh RM on newly generated candidates (matches paper's approach of continuously improving preference model)
- **Excel-only persistence**: No MLflow/W&B integration; lightweight pandas-based export suitable for toy-scale experiments
- **Sequential execution**: No parallelization across experiments (GPU memory constraint); experiments run one at a time with cleanup between
- **Checkpoint autosave**: Every 5 experiments to enable resume on Colab disconnects
