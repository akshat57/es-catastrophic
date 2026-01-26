import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 

import pandas as pd
import torch
import gc
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import re
import random
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import wandb

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    try:
        from torch.distributed._tensor import DTensor
    except ImportError:
        DTensor = None

random.seed(42)

# --- WandB Retrieval ---
def fetch_wandb_metrics():
    """
    Fetches countdown metrics from specified WandB runs.
    Returns dictionaries mapping step/iteration to scores for GRPO and ES.
    """
    api = wandb.Api()
    
    # GRPO Run
    grpo_scores = {}

    run = api.run("<wandb_organization>/<team>/<project>/<run_name>")
    history = run.history() # Fetch all columns
    max_step_with_score = -1
    
    for _, row in history.iterrows():
        # Try 'training/global_step' or 'global_step'
        step = row.get("training/global_step")
        if pd.isna(step):
            step = row.get("global_step")
        
        score = row.get("val-core/countdown_json/reward/mean@1")
        
        if pd.notna(step) and pd.notna(score):
            s = int(step)
            grpo_scores[s] = float(score)
            if s > max_step_with_score:
                max_step_with_score = s
                
    print(f"Fetched {len(grpo_scores)} GRPO scores. Max step: {max_step_with_score}")
    
    # Fallback for step 500 if missing but we have data near it (e.g. 490 or 498)
    if 500 not in grpo_scores and max_step_with_score > 400:
        print(f"Step 500 missing, using score from step {max_step_with_score}: {grpo_scores[max_step_with_score]}")
        grpo_scores[500] = grpo_scores[max_step_with_score]
        

    # ES Run
    es_scores = {}
    run = api.run("<wandb_organization>/<team>/<project>/<run_name>")
    history = run.history()
    for _, row in history.iterrows():
        if pd.notna(row["iteration"]) and pd.notna(row["val_acc"]):
            it = int(row["iteration"])
            score = float(row["val_acc"])
            es_scores[it] = score
    print(f"Fetched {len(es_scores)} ES scores.")
    
    # Save for debug
    with open("fetched_scores.json", "w") as f:
        json.dump({"grpo": grpo_scores, "es": es_scores}, f, indent=2)
        
    return grpo_scores, es_scores

# --- FSDP Loading Utilities ---
def load_fsdp_state_dict(checkpoint_dir):
    """
    Loads an FSDP checkpoint into a single state_dict (CPU).
    """
    checkpoint_path = Path(checkpoint_dir)
    config_path = checkpoint_path / "fsdp_config.json"
    
    if not config_path.exists():
        # Fallback: try to find world size from filenames if config is missing
        files = list(checkpoint_path.glob("model_world_size_*_rank_0.pt"))
        if not files:
            raise FileNotFoundError(f"No FSDP checkpoint files found in {checkpoint_dir}")
        match = re.search(r"model_world_size_(\d+)_rank_0\.pt", files[0].name)
        if match:
            world_size = int(match.group(1))
        else:
            raise ValueError(f"Could not determine world size from {files[0].name}")
    else:
        with open(config_path) as f:
            config = json.load(f)
        world_size = config["world_size"]

    print(f"Loading FSDP checkpoint from {checkpoint_dir} (World Size: {world_size})...")

    # Helper to load a single shard
    def load_shard(rank):
        shard_path = checkpoint_path / f"model_world_size_{world_size}_rank_{rank}.pt"
        return torch.load(shard_path, map_location="cpu", weights_only=False)

    # Parallel load shards
    shards = [None] * world_size
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
        futures = {executor.submit(load_shard, rank): rank for rank in range(world_size)}
        for future in tqdm(futures, desc="Loading Shards", total=world_size):
            rank = futures[future]
            shards[rank] = future.result()

    # Merge shards
    print("Merging shards...")
    state_dict = {}
    
    # We iterate over keys in the first shard (rank 0). 
    # Note: FSDP shards usually have the same keys, but some might be sharded (DTensor) and some replicated (Tensor).
    # However, with DTensor, the key exists in all shards?
    # Let's assume consistent keys across shards for FSDP.
    
    keys = list(shards[0].keys())
    
    for key in tqdm(keys, desc="Merging Keys"):
        # Check type of first shard's value
        val_0 = shards[0][key]
        
        if isinstance(val_0, DTensor):
            placements = tuple(val_0.placements)
            is_sharded = any(p.is_shard() for p in placements)
            
            if is_sharded:
                shard_dim = next(p.dim for p in placements if p.is_shard())
                local_tensors = []
                for rank in range(world_size):
                    # verify placement consistency?
                    t = shards[rank][key]
                    if isinstance(t, DTensor):
                        local_tensors.append(t._local_tensor)
                    else:
                        local_tensors.append(t) # Should not happen if inconsistent
                
                merged_tensor = torch.cat(local_tensors, dim=shard_dim)
                state_dict[key] = merged_tensor
            else:
                # Replicated
                state_dict[key] = val_0._local_tensor
        else:
            state_dict[key] = val_0

    # Clean keys
    print("Cleaning keys...")
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("_fsdp_wrapped_module.", "")
        new_state_dict[new_k] = v
        
    return new_state_dict


# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints/countdown/Qwen_qwen2.5-1.5B_Instruct_es_random_seed33_pop30_fp16_1024_tokens_200_samples_template_norm/Qwen")
# Helper to construct path
def get_model_path(iter_name):
    return os.path.join(CHECKPOINT_DIR, f"Qwen2.5-1.5B-Instruct_es_random_seed33_pop30_{iter_name}_sigma0.001_alpha0.0005_fp16_procs4_question_num200_checkpoint")

ES_MODELS_TO_EVALUATE = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "iter50": get_model_path("iter50"),
    "iter100": get_model_path("iter100"),
    "iter150": get_model_path("iter150"),
    "iter200": get_model_path("iter200"),
    "iter250": get_model_path("iter250"),
    "iter300": get_model_path("iter300"),
    "iter350": get_model_path("iter350"),
    "iter400": get_model_path("iter400"),
    "iter450": get_model_path("iter450"),
    "iter500": os.path.join(CHECKPOINT_DIR, "Qwen2.5-1.5B-Instruct_es_random_seed33_pop30_iter500_sigma0.001_alpha0.0005_fp16_procs4_question_num200_final")
}

GRPO_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "converted_checkpoints")
def get_grpo_model_path(step_name):
    # Return the directory containing the converted checkpoint
    return os.path.join(GRPO_CHECKPOINT_DIR, step_name)

GRPO_MODELS_TO_EVALUATE = {
    "grpo_step30": get_grpo_model_path("grpo_step30"),
    "grpo_step60": get_grpo_model_path("grpo_step60"),
    "grpo_step90": get_grpo_model_path("grpo_step90"),
    "grpo_step120": get_grpo_model_path("grpo_step120"),
    "grpo_step150": get_grpo_model_path("grpo_step150"),
    "grpo_step180": get_grpo_model_path("grpo_step180"),
    "grpo_step210": get_grpo_model_path("grpo_step210"),
    "grpo_step240": get_grpo_model_path("grpo_step240"),
    "grpo_step270": get_grpo_model_path("grpo_step270"),
    "grpo_step300": get_grpo_model_path("grpo_step300"),
    "grpo_step330": get_grpo_model_path("grpo_step330"),
    "grpo_step360": get_grpo_model_path("grpo_step360"),
    "grpo_step390": get_grpo_model_path("grpo_step390"),
    "grpo_step420": get_grpo_model_path("grpo_step420"),
    "grpo_step450": get_grpo_model_path("grpo_step450"),
    "grpo_step480": get_grpo_model_path("grpo_step480"),
    "grpo_step500": get_grpo_model_path("grpo_step500")
}

DATASET_PATHS = {
    "countdown": os.path.join(PROJECT_ROOT, "countdown/data/test.parquet")
}
TASKS = ["countdown", "hellaswag"]
CALC_FROBENIUS = True
CALC_KL = True
GPU_UTILIZATION = 0.8
CSV_OUTPUT_PATH = "results/es_evaluation_metrics.csv"

# --- 1. Load Datasets ---
print("Loading datasets...")
try:
    countdown_df = pd.read_parquet(DATASET_PATHS["countdown"])
    print("Countdown dataset loaded.")
    print(f"Countdown examples: {len(countdown_df)}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# --- Load Tokenizer & Template ---
print("Loading tokenizer and chat template...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
try:
    # Use chat template from iter100 if available, else default
    template_path = os.path.join(ES_MODELS_TO_EVALUATE["iter500"], "chat_template.jinja")
    if os.path.exists(template_path):
        print(template_path)
        with open(template_path, "r") as f:
            chat_template = f.read()
        print("Chat template loaded from iter100.")
    else:
        # Fallback to base tokenizer's chat template or a default
        print("Warning: chat_template.jinja not found. Using tokenizer.chat_template.")
        chat_template = tokenizer.chat_template
except Exception:
    print("Warning: Error loading chat_template.jinja. Using tokenizer.chat_template.")
    chat_template = tokenizer.chat_template


# --- Helper Functions for HellaSwag ---
def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("[[.*?]]", "", text)
    text = text.replace("  ", " ")
    return text

def process_docs(dataset, eval_type='full'):
    def _process_doc(doc):
        if eval_type == 'full':
            ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        else:
            ctx = doc["ctx_b"].capitalize()
        choices = [preprocess(ending) for ending in doc["endings"]]
        gold_index = int(doc["label"])

        indices = list(range(len(choices)))
        random.shuffle(indices)
        shuffled_choices = [choices[i] for i in indices]
        new_gold_index = indices.index(gold_index)

        if eval_type == 'full':
            query = preprocess(doc["activity_label"] + ": " + ctx)
        else:
            query = preprocess(ctx)

        out_doc = {
            "query": query,
            "choices": shuffled_choices,
            "gold": new_gold_index,
        }
        return out_doc

    return dataset.map(_process_doc)

def construct_prompt(context, endings):
    prompt = (
        "You are given a situation followed by four possible endings. "
        "Choose the most appropriate ending by selecting the corresponding number. "
        "Respond only with the number of the correct answer.\n\n"
        f"Context: {context}\n"
    )
    for i, ending in enumerate(endings):
        prompt += f"{i + 1}. {ending}\n"
    prompt += "\nAnswer: "
    return prompt

# --- 3. Evaluation Functions ---
import sys
# sys.path.append(os.path.join(PROJECT_ROOT, "countdown"))
# from countdown_task import reward_function
from countdown_reward import reward_function

def evaluate_countdown(llm, df, tokenizer, return_data=False):
    """Evaluates a model on the Countdown dataset."""
    df = df.head(500)
    prompts_raw = df['context'].tolist()
    numbers_list = df['numbers'].tolist()
    targets_list = df['target'].tolist()
    
    prompts = []
    for p in prompts_raw:
        msgs = [{"role": "user", "content": p}]
        s = tokenizer.apply_chat_template(
            msgs, 
            tokenize=False, 
            add_generation_prompt=True,
            chat_template=chat_template
        )
        prompts.append(s)
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, seed=42, logprobs=1 if return_data else None)
    outputs = llm.generate(prompts, sampling_params)
    
    predictions = [o.outputs[0].text for o in outputs]
    
    correct = 0
    generated_data = []
    
    for i, (pred, numbers, target) in enumerate(zip(predictions, numbers_list, targets_list)):
        res = reward_function(response=pred, numbers=numbers, target=target)
        if res["reward"] >= 1.0:
            correct += 1
        
        if return_data:
            output_item = outputs[i]
            processed_logprobs = []
            if output_item.outputs[0].logprobs is not None: 
                for logprob_dict in output_item.outputs[0].logprobs:
                    processed_dict = {token_id: lp.logprob for token_id, lp in logprob_dict.items()}
                    processed_logprobs.append(processed_dict)
            generated_data.append({
                "prompt_text": prompts[i],
                "prompt_token_ids": output_item.prompt_token_ids,
                "generated_text": pred,
                "generated_token_ids": output_item.outputs[0].token_ids,
                "logprobs": processed_logprobs 
            })
            
    accuracy = correct / len(df)
    return accuracy, generated_data

def evaluate_hellaswag(llm, tokenizer, return_data=False):
    """Evaluates a model on the Hellaswag dataset using custom logic."""
    try:
        dataset = load_dataset("hellaswag", split="validation")
        dataset = dataset.select(range(0, len(dataset), 10))
        
        eval_type = 'full'
        dataset = process_docs(dataset, eval_type=eval_type)
        
        prompts = []
        golds = []
        
        for example in dataset:
            context = example["query"]
            endings = example["choices"]
            correct_answer_idx = example["gold"]
            
            raw_prompt = construct_prompt(context, endings)
            
            msgs = [{"role": "user", "content": raw_prompt}]
            templated_prompt = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=chat_template
            )
            
            prompts.append(templated_prompt)
            golds.append(correct_answer_idx)

        sampling_params = SamplingParams(temperature=0.0, max_tokens=10, logprobs=1 if return_data else None)
        outputs = llm.generate(prompts, sampling_params)
        
        correct = 0
        total = len(prompts)
        generated_data = []
        
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            digits = [a for a in generated_text if a in '1234']
            
            if len(digits) > 0:
                predicted_index = int(digits[0]) - 1 
                if predicted_index == golds[i]:
                    correct += 1
            
            if return_data:
                output_item = outputs[i]
                processed_logprobs = []
                if output_item.outputs[0].logprobs is not None:
                    for logprob_dict in output_item.outputs[0].logprobs:
                        processed_dict = {token_id: lp.logprob for token_id, lp in logprob_dict.items()}
                        processed_logprobs.append(processed_dict)
                generated_data.append({
                    "prompt_text": prompts[i],
                    "prompt_token_ids": output_item.prompt_token_ids,
                    "generated_text": output.outputs[0].text,
                    "generated_token_ids": output_item.outputs[0].token_ids,
                    "logprobs": processed_logprobs
                })
            
        accuracy = correct / total
        return accuracy, generated_data
        
    except Exception as e:
        print(f"Error evaluating Hellaswag: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, []

# --- MMLU Helpers ---
choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def evaluate_mmlu(llm, tokenizer, n_shots=5):
    """
    Evaluates a model on the MMLU dataset.
    Samples 500 random examples across all tasks.
    """
    print("Loading MMLU dataset...")
    try:
        # Load all MMLU data
        ds = load_dataset("cais/mmlu", "all")
        all_test_examples = []
        all_dev_examples = {} # Map subject -> dev_df for few-shot
        
        if 'test' in ds:
            test_data = ds['test']
            dev_data = ds['dev']
        else:
            test_data = ds['test']
            dev_data = ds['dev']

        # Sample 500 random indices
        total_test = len(test_data)
        indices = random.sample(range(total_test), min(500, total_test))
        
        dev_df_all = dev_data.to_pandas()
        
        correct = 0
        total = 0
        
        prompts = []
        labels = []
        
        print(f"Preparing {len(indices)} MMLU examples...")
        
        test_df_all = test_data.to_pandas()
        
        for idx in indices:
            row = test_df_all.iloc[idx]
            subject = row['subject'] if 'subject' in row else 'General Knowledge'
            
            # Get few-shot examples for this subject
            # Filter dev_df for this subject
            if 'subject' in dev_df_all.columns:
                subject_dev_df = dev_df_all[dev_df_all['subject'] == subject]
            else:
                subject_dev_df = dev_df_all # Fallback
            
            # If subject_dev_df is empty or small, take what we can
            k = min(n_shots, len(subject_dev_df))
            
            prompt_header = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
            
            examples_text = ""
            for i in range(k):
                dev_row = subject_dev_df.iloc[i]
                q = dev_row['question']
                c = dev_row['choices'] # List of 4 strings
                a_idx = dev_row['answer'] # Int 0-3
                a_char = choices[a_idx]
                
                examples_text += f"{q}\n"
                for j, choice in enumerate(c):
                    examples_text += f"{choices[j]}. {choice}\n"
                examples_text += f"Answer: {a_char}\n\n"
                
            # Current Question
            q = row['question']
            c = row['choices']
            a_idx = row['answer']
            a_char = choices[a_idx]
            
            query_text = f"{q}\n"
            for j, choice in enumerate(c):
                query_text += f"{choices[j]}. {choice}\n"
            query_text += "Answer:"
            
            full_prompt_raw = prompt_header + examples_text + query_text
            
            msgs = [{"role": "user", "content": full_prompt_raw}]
            
            templated_prompt = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=chat_template
            )
            
            prompts.append(templated_prompt)
            labels.append(a_idx) # 0, 1, 2, 3
            
        # Generate
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        outputs = llm.generate(prompts, sampling_params)
        
        # Score
        for i, output in enumerate(outputs):
            logprobs = output.outputs[0].logprobs[0] # Dict of {token_id: logprob}
            
            token_map = {} 
            for char in ["A", "B", "C", "D"]:
                # Try raw
                tid = tokenizer.encode(char, add_special_tokens=False)[0] # Usually single token
                token_map[char] = [tid]
                
                # Try with space
                tid_space = tokenizer.encode(" " + char, add_special_tokens=False)[0]
                token_map[char].append(tid_space)
                
            # Find best char
            best_char = -1
            max_lp = -float('inf')
            
            for idx, char in enumerate(["A", "B", "C", "D"]):
                lp = -float('inf')
                for tid in token_map[char]:
                    if tid in logprobs:
                        lp = max(lp, logprobs[tid].logprob)
                
                if lp > max_lp:
                    max_lp = lp
                    best_char = idx
            
            if best_char == labels[i]:
                correct += 1
            total += 1
            
        acc = correct / total
        return acc

    except Exception as e:
        print(f"Error evaluating MMLU: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def compute_sequence_logprobs(llm, sequence_data):
    """
    Computes log probabilities for a list of sequences (prompt + generation).
    sequence_data: List of dicts with 'prompt_token_ids' and 'generated_token_ids'.
    Returns: List of sum(logprobs) for the generated part.
    """
    full_prompts_token_ids = []
    generation_lengths = []
    
    for item in sequence_data:
        full_ids = item['prompt_token_ids'] + item['generated_token_ids']
        full_prompts_token_ids.append(full_ids)
        generation_lengths.append(len(item['generated_token_ids']))
        
    # prompt_logprobs=20 to catch the actual token if possible
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=20)
    
    # Wrap token IDs in TokensPrompt-like dicts
    inputs = [{"prompt_token_ids": ids} for ids in full_prompts_token_ids]
    
    # Pass wrapped inputs
    outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)
    
    seq_logprobs = []
    for i, output in enumerate(outputs):
        gen_len = generation_lengths[i]
        if gen_len == 0:
            seq_logprobs.append(0.0)
            continue
        logprobs_list = output.prompt_logprobs
        relevant_logprobs_dicts = logprobs_list[-gen_len:]
        target_ids = sequence_data[i]['generated_token_ids']
        
        current_seq_log_sum = 0.0
        
        for j, token_logprob_dict in enumerate(relevant_logprobs_dicts):
            if token_logprob_dict is None:
                continue 
            
            tid = target_ids[j]
            if tid in token_logprob_dict:
                current_seq_log_sum += token_logprob_dict[tid].logprob
            else:
                # Token not in top-K. Use a low value or the min of returned.
                if token_logprob_dict:
                    min_lp = min(v.logprob for v in token_logprob_dict.values())
                    current_seq_log_sum += (min_lp - 2.0)
                else:
                    current_seq_log_sum += -15.0 
                
        seq_logprobs.append(current_seq_log_sum)
        
    return seq_logprobs

def extract_logprobs_from_gen(gen_data):
    """
    Extracts the sum of logprobs from the generation output (P).
    gen_data: List of dicts from evaluate functions.
    """
    log_sums = []
    for item in gen_data:
        curr_sum = 0.0
        gen_ids = item['generated_token_ids']
        
        # Handle empty generation
        if not gen_ids:
            log_sums.append(0.0)
            continue

        for j, lg_dict in enumerate(item['logprobs']):
            if j >= len(gen_ids): break
            tid = gen_ids[j]
            if tid in lg_dict:
                curr_sum += lg_dict[tid]
        log_sums.append(curr_sum)
    return log_sums


# KL Divergence and frobenius calculation


def compute_frobenius_norm(results, base_model_id, models_to_evaluate):
    """
    Computes Frobenius norm between each model and the base model.
    Updates results in place.
    
    :param results: Dictionary of results from evaluations.
    :param base_model_id: Base model ID for scoring.
    :param models_to_evaluate: Dictionary mapping model names to paths.
    """
    print("\n--- Computing Frobenius Norms ---")
    print("Loading base model weights...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="cpu")
        base_sd = base_model.state_dict()
        del base_model
        gc.collect()
    except Exception as e:
        print(f"Error loading base model for norm calculation: {e}")
        base_sd = None

    if base_sd is not None:
        for model_name, model_path in models_to_evaluate.items():
            # Only compute for models present in results
            if model_name not in results:
                continue

            if model_name == "base_model":
                results[model_name]["frobenius_norm"] = 0.0
                continue
            
            if not os.path.exists(model_path):
                results[model_name]["frobenius_norm"] = None
                continue

            print(f"Computing norm for {model_name}...")
            try:
                curr_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cpu")
                curr_sd = curr_model.state_dict()
                
                total_sq_diff = 0.0
                for key in base_sd:
                    if key in curr_sd:
                        diff = base_sd[key] - curr_sd[key]
                        total_sq_diff += torch.sum(diff ** 2).item()
                
                norm = np.sqrt(total_sq_diff)
                results[model_name]["frobenius_norm"] = norm
                print(f"  Norm: {norm:.4f}")
                
                del curr_model, curr_sd
                gc.collect()
                
            except Exception as e:
                print(f"Error computing norm for {model_name}: {e}")
                results[model_name]["frobenius_norm"] = None
    
    if base_sd is not None:
        del base_sd
        gc.collect()


def compute_kl_divergence(results, base_model_id, gpu_utilization=0.8):
    """
    Computes KL divergence between each model and the base model given generation data.
    Then updates results in place.
    
    :param results: Dictionary of results from evaluations.
    :param base_model_id: Base model ID for scoring.
    :param gpu_utilization: GPU memory utilization for LLM.
    """
    print("\n--- Computing KL Divergence ---")
    print(f"Loading Base Model ({base_model_id}) for scoring...")
    llm_base = LLM(model=base_model_id, 
                   trust_remote_code=True, 
                   gpu_memory_utilization=gpu_utilization, 
                   tensor_parallel_size=torch.cuda.device_count())
    
    for model_name, res in results.items():
        if model_name == "base_model":
            res["kl_countdown"] = 0.0
            res["kl_hellaswag"] = 0.0
            continue
            
        print(f"Computing KL for {model_name}...")
        
        if "countdown_data" in res:
            gen_data = res["countdown_data"]
            log_p = extract_logprobs_from_gen(gen_data)
            log_q = compute_sequence_logprobs(llm_base, gen_data)
            kls = [p - q for p, q in zip(log_p, log_q)]
            res["kl_countdown"] = float(np.mean(kls))
            print(f"  KL Countdown: {res['kl_countdown']:.4f}")
            del res["countdown_data"]
        if "hellaswag_data" in res:
            gen_data = res["hellaswag_data"]
            log_p = extract_logprobs_from_gen(gen_data)
            log_q = compute_sequence_logprobs(llm_base, gen_data)
            kls = [p - q for p, q in zip(log_p, log_q)]
            res["kl_hellaswag"] = float(np.mean(kls))
            print(f"  KL Hellaswag: {res['kl_hellaswag']:.4f}")
            del res["hellaswag_data"]
    
    del llm_base
    gc.collect()
    torch.cuda.empty_cache()

# --- Plotting Functions ---
import seaborn as sns
sns.set_style("darkgrid")

def plot_iter_vs_kl(df, metric_prefix="es"):
    kl_cd_col = f"{metric_prefix}-countdown-kl"
    kl_hs_col = f"{metric_prefix}-hellaswag-kl"
    
    if "iteration" in df.columns and (kl_cd_col in df.columns or kl_hs_col in df.columns):
        plt.figure(figsize=(10, 6))
        if kl_cd_col in df.columns:
            sns.lineplot(data=df, x="iteration", y=kl_cd_col, marker='o', label='Countdown KL')
        if kl_hs_col in df.columns:
            sns.lineplot(data=df, x="iteration", y=kl_hs_col, marker='X', label='Hellaswag KL') # using 'X' as close approx for 'x'
        plt.xlabel("Iteration")
        plt.ylabel("KL Divergence from Base")
        plt.title(f"Iteration vs KL Divergence ({metric_prefix})")
        plt.legend()
        plt.grid(True)
        filename = f"results/iter_vs_kl_{metric_prefix}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

def plot_countdown_score_vs_kl(df, metric_prefix="es"):
    kl_cd_col = f"{metric_prefix}-countdown-kl"
    score_cd_col = f"{metric_prefix}-countdown-score"

    if kl_cd_col in df.columns and score_cd_col in df.columns:
        plt.figure(figsize=(10, 6))

        ax = sns.scatterplot(
            data=df,
            x=kl_cd_col,
            y=score_cd_col,
        )

        # Annotate points with iteration
        if "iteration" in df.columns:
            for _, row in df.iterrows():
                # Check for NaN before annotating
                if pd.notna(row[kl_cd_col]) and pd.notna(row[score_cd_col]):
                    ax.annotate(
                        f"iter{int(row['iteration'])}",
                        (row[kl_cd_col], row[score_cd_col]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        ha="left",
                        fontsize=9,
                    )

        ax.set_xlabel("KL Divergence")
        ax.set_ylabel("Countdown")
        ax.set_title(f"Countdown Score vs KL ({metric_prefix})")
        ax.grid(True)

        plt.tight_layout()
        filename = f"results/countdown_score_vs_kl_{metric_prefix}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

def plot_hellaswag_score_vs_kl(df, metric_prefix="es"):
    kl_hs_col = f"{metric_prefix}-hellaswag-kl"
    score_hs_col = f"{metric_prefix}-hellaswag-score"

    if kl_hs_col in df.columns and score_hs_col in df.columns:
        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(
            data=df,
            x=kl_hs_col,
            y=score_hs_col,
        )

        # Annotate points with iteration
        if "iteration" in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row[kl_hs_col]) and pd.notna(row[score_hs_col]):
                    ax.annotate(
                        f"iter{int(row['iteration'])}",
                        (row[kl_hs_col], row[score_hs_col]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        ha="left",
                        fontsize=9,
                    )

        ax.set_xlabel("KL Divergence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Hellaswag Score vs KL ({metric_prefix})")
        ax.grid(True)

        plt.tight_layout()
        filename = f"results/hellaswag_score_vs_kl_{metric_prefix}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

def plot_mmlu_score_vs_kl(df, metric_prefix="es"):
    kl_hs_col = f"{metric_prefix}-hellaswag-kl" # Use Hellaswag KL as proxy for general KL
    score_mmlu_col = f"{metric_prefix}-mmlu-score"

    if kl_hs_col in df.columns and score_mmlu_col in df.columns:
        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(
            data=df,
            x=kl_hs_col,
            y=score_mmlu_col,
        )

        # Annotate points with iteration
        if "iteration" in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row[kl_hs_col]) and pd.notna(row[score_mmlu_col]):
                    ax.annotate(
                        f"iter{int(row['iteration'])}",
                        (row[kl_hs_col], row[score_mmlu_col]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        ha="left",
                        fontsize=9,
                    )

        ax.set_xlabel("KL Divergence (Hellaswag)")
        ax.set_ylabel("MMLU Accuracy")
        ax.set_title(f"MMLU Score vs KL ({metric_prefix})")
        ax.grid(True)

        plt.tight_layout()
        filename = f"results/mmlu_score_vs_kl_{metric_prefix}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

def plot_iter_vs_frobenius(df, metric_prefix="es"):
    frob_col = f"{metric_prefix}-frobenius-norm"

    if "iteration" in df.columns and frob_col in df.columns:
        # Filter out NaN values
        plot_df = df.dropna(subset=[frob_col])
        if not plot_df.empty:
            plt.figure(figsize=(10, 6))

            ax = sns.lineplot(
                data=plot_df,
                x="iteration",
                y=frob_col,
                marker="o",
            )

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Frobenius Norm")
            ax.set_title(f"Iteration vs Frobenius Norm from Base Model ({metric_prefix})")
            ax.grid(True)

            plt.tight_layout()
            filename = f"results/iter_vs_frobenius_{metric_prefix}.png"
            plt.savefig(filename)
            print(f"Saved {filename}")
            plt.close()

def plot_countdown_score_vs_frobenius(df, metric_prefix="es"):
    frob_col = f"{metric_prefix}-frobenius-norm"
    score_cd_col = f"{metric_prefix}-countdown-score"

    if frob_col in df.columns and score_cd_col in df.columns:
        plot_df = df.dropna(subset=[frob_col, score_cd_col])
        if not plot_df.empty:
            plt.figure(figsize=(10, 6))

            ax = sns.scatterplot(
                data=plot_df,
                x=frob_col,
                y=score_cd_col,
                marker="o",
            )
            # Annotate points with iteration
            if "iteration" in plot_df.columns:
                for _, row in plot_df.iterrows():
                    ax.annotate(
                        f"iter{int(row['iteration'])}",
                        (row[frob_col], row[score_cd_col]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        ha="left",
                        fontsize=9,
                    )
            ax.set_xlabel("Frobenius Norm")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Countdown Score vs Frobenius Norm ({metric_prefix})")
            ax.grid(True)
            
            plt.tight_layout()
            filename = f"results/countdown_score_vs_frobenius_{metric_prefix}.png"
            plt.savefig(filename)
            print(f"Saved {filename}")
            plt.close()

def plot_hellaswag_score_vs_frobenius(df, metric_prefix="es"):
    frob_col = f"{metric_prefix}-frobenius-norm"
    score_hs_col = f"{metric_prefix}-hellaswag-score"

    if frob_col in df.columns and score_hs_col in df.columns:
        plot_df = df.dropna(subset=[frob_col, score_hs_col])
        if not plot_df.empty:
            plt.figure(figsize=(10, 6))

            ax = sns.scatterplot(
                data=plot_df,
                x=frob_col,
                y=score_hs_col,
                marker="o",
                color='green'
            )
            # Annotate points with iteration
            if "iteration" in plot_df.columns:
                for _, row in plot_df.iterrows():
                    ax.annotate(
                        f"iter{int(row['iteration'])}",
                        (row[frob_col], row[score_hs_col]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        ha="left",
                        fontsize=9,
                    )
            ax.set_xlabel("Frobenius Norm")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Hellaswag Score vs Frobenius Norm ({metric_prefix})")
            ax.grid(True)
            
            plt.tight_layout()
            filename = f"results/hellaswag_score_vs_frobenius_{metric_prefix}.png"
            plt.savefig(filename)
            print(f"Saved {filename}")
            plt.close()

def plot_mmlu_score_vs_frobenius(df, metric_prefix="es"):
    frob_col = f"{metric_prefix}-frobenius-norm"
    score_mmlu_col = f"{metric_prefix}-mmlu-score"

    if frob_col in df.columns and score_mmlu_col in df.columns:
        plot_df = df.dropna(subset=[frob_col, score_mmlu_col])
        if not plot_df.empty:
            plt.figure(figsize=(10, 6))

            ax = sns.scatterplot(
                data=plot_df,
                x=frob_col,
                y=score_mmlu_col,
                marker="o",
                color='red'
            )
            # Annotate points with iteration
            if "iteration" in plot_df.columns:
                for _, row in plot_df.iterrows():
                    ax.annotate(
                        f"iter{int(row['iteration'])}",
                        (row[frob_col], row[score_mmlu_col]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        ha="left",
                        fontsize=9,
                    )
            ax.set_xlabel("Frobenius Norm")
            ax.set_ylabel("MMLU Accuracy")
            ax.set_title(f"MMLU Score vs Frobenius Norm ({metric_prefix})")
            ax.grid(True)
            
            plt.tight_layout()
            filename = f"results/mmlu_score_vs_frobenius_{metric_prefix}.png"
            plt.savefig(filename)
            print(f"Saved {filename}")
            plt.close()

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["countdown", "hellaswag"], 
                        choices=["countdown", "hellaswag"], help="Tasks to run")
    parser.add_argument("--es_models_to_run", nargs="*", default=None,
                        help="List of ES models to evaluate. If not provided, runs ALL ES models. If provided empty, runs none.")
    parser.add_argument("--grpo_models_to_run", nargs="*", default=None,
                        help="List of GRPO models to evaluate. If not provided, runs ALL GRPO models. If provided empty, runs none.")
    parser.add_argument("--calc_frobenius", action=argparse.BooleanOptionalAction, default=True, help="Compute Frobenius norms")
    parser.add_argument("--calc_kl", action=argparse.BooleanOptionalAction, default=True, help="Compute KL divergence")
    parser.add_argument("--gpu_utilization", type=float, default=0.8, help="GPU memory utilization for LLM")
    parser.add_argument("--evaluate_mmlu", action=argparse.BooleanOptionalAction, default=False, help="Evaluate on MMLU (500 samples)")
    args = parser.parse_args()
    
    calc_kl = args.calc_kl
    
    # --- Fetch WandB Metrics ---
    grpo_scores, es_scores = fetch_wandb_metrics()
    
    results = {}

    # Build the full registry of models
    ALL_MODELS = ES_MODELS_TO_EVALUATE.copy()
    ALL_MODELS.update(GRPO_MODELS_TO_EVALUATE)
    
    # Determine the actual list of models to run based on args
    models_to_run = []

    # ES Logic
    if args.es_models_to_run is None:
        # Flag not provided -> Run ALL ES models
        models_to_run.extend(ES_MODELS_TO_EVALUATE.keys())
    else:
        # Flag provided (possibly empty) -> Run specified
        models_to_run.extend(args.es_models_to_run)

    # GRPO Logic
    if args.grpo_models_to_run is None:
        # Flag not provided -> Run ALL GRPO models
        models_to_run.extend(GRPO_MODELS_TO_EVALUATE.keys())
    else:
        # Flag provided (possibly empty) -> Run specified
        models_to_run.extend(args.grpo_models_to_run)
    
    evaluated_model_paths = {}

    def get_iter(name):
        if "base" in name: return 0
        m = re.search(r'(iter|step)(\d+)', name)
        return int(m.group(2)) if m else -1

    # 1. Evaluate Models
    evaluated_model_paths = {}
    for model_name, model_path in ALL_MODELS.items():
        if model_name not in models_to_run:
            # print(f"Skipping {model_name} as it's not in models_to_run")
            continue
        print(f"--- Evaluating {model_name} ---")
        
        is_grpo = "grpo" in model_name
        
        if not os.path.exists(model_path) and model_name != "base_model":
            print(f"Skipping {model_name} (path not found)")
            continue
            
        final_model_path = model_path
        
        if is_grpo:
            # Convert FSDP to HF format if needed
            converted_dir = Path("converted_checkpoints") / model_name
            if converted_dir.exists() and (converted_dir / "config.json").exists():
                print(f"Found converted checkpoint at {converted_dir}")
                final_model_path = str(converted_dir)
            else:
                print(f"Converting FSDP weights for {model_name}...")
                try:
                    converted_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Load base model config
                    print(f"Loading base model config from {ES_MODELS_TO_EVALUATE['base_model']}...")
                    base_model = AutoModelForCausalLM.from_pretrained(
                        ES_MODELS_TO_EVALUATE["base_model"], 
                        torch_dtype=torch.float16, 
                        device_map="cpu",
                        trust_remote_code=True
                    )
                    
                    # Load FSDP weights
                    state_dict = load_fsdp_state_dict(model_path)
                    
                    # Load weights into base model
                    print("Loading weights into base model...")
                    missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
                    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                    
                    # Save to converted_dir
                    print(f"Saving converted model to {converted_dir}...")
                    base_model.save_pretrained(converted_dir)
                    tokenizer.save_pretrained(converted_dir)
                    
                    del base_model
                    del state_dict
                    gc.collect()
                    final_model_path = str(converted_dir)
                    
                except Exception as e:
                    print(f"Error converting GRPO weights: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        evaluated_model_paths[model_name] = final_model_path

        print(f"Initializing vLLM with {final_model_path}...")
        try:
            llm = LLM(model=final_model_path, trust_remote_code=True, gpu_memory_utilization=0.8, tensor_parallel_size=torch.cuda.device_count())
        except Exception as e:
            print(f"Error initializing vLLM: {e}")
            continue
        
        model_results = {}
        
        if "countdown" in args.tasks or calc_kl: 
            print("Evaluating Countdown...")
            local_acc, gen_data = evaluate_countdown(llm, countdown_df, tokenizer, return_data=calc_kl)
            
            # Override with WandB score if available
            iteration = get_iter(model_name)
            acc = local_acc
            # if is_grpo and iteration in grpo_scores:
            #     print(f"Overriding local Countdown Acc ({local_acc:.4f}) with WandB score: {grpo_scores[iteration]}")
            #     acc = grpo_scores[iteration]
            # elif not is_grpo and iteration in es_scores:
            #      # Optional: Override ES scores too if needed, but user specifically asked for GRPO correction
            #      # print(f"Overriding local Countdown Acc ({local_acc:.4f}) with ES WandB score: {es_scores[iteration]}")
            #      # acc = es_scores[iteration]
            #      pass

            model_results["countdown_accuracy"] = acc
            if calc_kl:
                model_results["countdown_data"] = gen_data
            print(f"Countdown Acc: {acc}")

        if "hellaswag" in args.tasks or calc_kl:
            print("Evaluating Hellaswag...")

            acc, gen_data = evaluate_hellaswag(llm, tokenizer, return_data=calc_kl)
            model_results["hellaswag_accuracy"] = acc
            if calc_kl:
                model_results["hellaswag_data"] = gen_data
            print(f"Hellaswag Acc: {acc}")
            
        if args.evaluate_mmlu:
            print("Evaluating MMLU...")
            acc = evaluate_mmlu(llm, tokenizer)
            model_results["mmlu_accuracy"] = acc
            print(f"MMLU Acc: {acc}")
            
        results[model_name] = model_results
        
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
    # 2. Compute KL Divergence if requested
    if calc_kl:
        compute_kl_divergence(results, base_model_id=ES_MODELS_TO_EVALUATE["base_model"], gpu_utilization=args.gpu_utilization)

    # --- Compute Frobenius Norm ---
    if args.calc_frobenius:
        compute_frobenius_norm(results, ES_MODELS_TO_EVALUATE["base_model"], evaluated_model_paths)

    # 3. Save Results
    print("\n--- Saving Results ---")
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Filter results to only include models_to_run to avoid artifacts
    results = {k: v for k, v in results.items() if k in models_to_run}

    with open("results/evaluation_results_kl.json", "w") as f:
        json.dump(results, f, indent=2)
        
    def get_iter(name):
        if "base" in name: return 0
        m = re.search(r'(iter|step)(\d+)', name)
        return int(m.group(2)) if m else -1
        
    sorted_names = sorted([n for n in results.keys() if "base" in n or "iter" in n or "step" in n], key=get_iter)

    # --- Save CSV for extensibility ---
    rows_by_iter = {}
    for name in sorted_names:
        it = get_iter(name)
        res = results[name]
        
        if "grpo" in name:
            metric_prefix = "grpo"
        else:
            metric_prefix = "es"

        if it not in rows_by_iter:
            rows_by_iter[it] = {"iteration": it}
        
        rows_by_iter[it].update({
            f"{metric_prefix}-countdown-score": res.get("countdown_accuracy", None),
            f"{metric_prefix}-hellaswag-score": res.get("hellaswag_accuracy", None),
            f"{metric_prefix}-mmlu-score": res.get("mmlu_accuracy", None),
            f"{metric_prefix}-countdown-kl": res.get("kl_countdown", None),
            f"{metric_prefix}-hellaswag-kl": res.get("kl_hellaswag", None),
            f"{metric_prefix}-frobenius-norm": res.get("frobenius_norm", None),
        })
    
    csv_rows = [rows_by_iter[it] for it in sorted(rows_by_iter.keys())]
    csv_df = pd.DataFrame(csv_rows)
    csv_output_path = "results/es_evaluation_metrics.csv"
    csv_df.to_csv(csv_output_path, index=False)
    print(f"Saved {csv_output_path}")

    # 4. Plotting from CSV
    print("\n--- Generating Plots from CSV ---")
    try:
        plot_df = pd.read_csv(csv_output_path)
        
        for ft_type in ["es", "grpo"]:
            plot_iter_vs_kl(plot_df, ft_type)
            plot_countdown_score_vs_kl(plot_df, ft_type)
            plot_hellaswag_score_vs_kl(plot_df, ft_type)
            plot_mmlu_score_vs_kl(plot_df, ft_type)
            plot_iter_vs_frobenius(plot_df, ft_type)
            plot_countdown_score_vs_frobenius(plot_df, ft_type)
            plot_hellaswag_score_vs_frobenius(plot_df, ft_type)
            plot_mmlu_score_vs_frobenius(plot_df, ft_type)
        
    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()
