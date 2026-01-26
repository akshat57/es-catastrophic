# ES Fine-Tuning Experiments

This repository contains implementations for fine-tuning Language Models using Evolution Strategies (ES), focusing on tasks like Countdown. It includes training scripts (vLLM/Ray and HF Accelerator), evaluation tools, and analysis scripts for sparsity.

## Installation
1.  **Environment Setup**:
    Ensure you have a Python environment (Python 3.10+ recommended).
2.  **Install Dependencies**:
    ```bash
    pip install -r requirement.txt
    ```

## 1. Training
**Script:** `countdown_run_vllm_template.py`

**Usage:**
```bash
python countdown_run_vllm_template.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --hf_cache_dir ./hf_cache \
  --precision fp16 \
  --num_processes 4 \
  --data_sample 200 \
  --num_iterations 500 \
  --task countdown \
  --reward_method norm \
  --apply_chat_template \
  --chat_template_file <path_to_template> \
  --eval_every 10 \
  --reward_fn_file countdown_reward.py \
  --reward_fn_name reward_function \
  --logs_dir logs \
  --ckpt_dir checkpoints \
  --save_model_every 50 \
  --wandb \
  --wandb_entity <your_entity> \
  --wandb_project <your_project> \
  --wandb_run_name <run_name> \
  --verbose
```

## 2. Evaluation

Evaluate trained checkpoints on specific tasks (e.g., Countdown, HellaSwag).

**Script:** `evaluation_script.py`

**Usage:**
```bash
python evaluation_script.py \
  --tasks countdown hellaswag \
  --es_models_to_run iter50 iter100 iter500 \
  --grpo_models_to_run grpo_step500 \
  --calc_frobenius \
  --calc_kl \
  --gpu_utilization 0.8 
```

*   `--es_models_to_run`: List specific ES iterations to evaluate (matches keys in the script's configuration).
*   `--grpo_models_to_run`: List specific GRPO steps to evaluate.
*   Results are saved to `results/es_evaluation_metrics.csv` and `results/evaluation_results_kl.json`.

**Interactive Notebook:** `evaluation_script.ipynb` mirrors the logic of the Python script for step-by-step execution and analysis.

## 3. Analysis & Visualization

### Sparsity Analysis
Calculate the sparsity of weight updates between a base model and a fine-tuned checkpoint.

**Script:** `visualize_update_sparsity.py`

**Usage:**
```bash
python visualize_update_sparsity.py \
  --before Qwen/Qwen2.5-1.5B-Instruct \
  --after checkpoints/my_finetuned_model \
  --tau 1e-4 \
  --json_out sparsity_report.json \
  --plot
```

### Comparison Plotting
To generate the comparison plot between ES and GRPO sparsity (as used in the paper/reports), you need to generate two specific JSON reports first:

1.  **Generate ES Report:**
    ```bash
    python visualize_update_sparsity.py --before <base_model> --after <es_checkpoint> --json_out sparsity_report_es.json
    ```

2.  **Generate GRPO Report:**
    ```bash
    python visualize_update_sparsity.py --before <base_model> --after <grpo_checkpoint> --json_out sparsity_report_grpo.json
    ```

3.  **Run Plotting Script:**
    ```bash
    python plot_sparsity_comparison.py
    ```
    This saves the plot to `results/sparsity_comparison.png`.

## Directory Structure

*   `countdown/`: Contains countdown-specific task logic and datasets.
    *   `countdown_task.py`: Defines reward functions.
    *   `data/`: Contains `countdown.json`, `train.parquet`, `test.parquet`.
*   `utils/`: Utility scripts (e.g., `worker_extn.py` for Ray/vLLM workers).
*   `previous_tasks/`: Legacy or reference task data (e.g., HellaSwag).