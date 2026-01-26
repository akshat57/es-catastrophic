# filename: es_countdown_parquet_reward_adapter.py
import os
# --- NCCL safer defaults (set before torch import) ---
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_TIMEOUT", "1800")  # seconds
# ------------------------------------------------------
import sys
import re
import json
import gc
import time
import argparse
import traceback
import functools
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

# ----------------- Reward adapter utils (from attached ref) -----------------
import inspect
import importlib.util, types

def _load_module_from_file(path: str) -> types.ModuleType:
    """Load a Python module from an arbitrary file path, without altering sys.path."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reward file not found: {path}")
    spec = importlib.util.spec_from_file_location("reward_mod", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def make_reward_adapters(file_path: str, fn_name: str = "reward_function"):
    mod = _load_module_from_file(file_path)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"{fn_name} in {file_path} is not callable")

    sig = inspect.signature(fn)
    fn_params = set(sig.parameters.keys())
    accepts_solution = "solution_str" in fn_params or "response" in fn_params
    solution_kw = "solution_str" if "solution_str" in fn_params else ("response" if "response" in fn_params else None)
    accepts_gt    = "ground_truth" in fn_params
    accepts_extra = "extra_info" in fn_params

    def _normalize_gt(v):
        if v is None:
            return None
        # compare as strings to avoid 1 vs 1.0 mismatches
        if isinstance(v, float):
            s = f"{v}"
            # strip trailing .0s
            return s.rstrip("0").rstrip(".") if "." in s else s
        return str(v)

    def _build_kwargs(pred_text: str, kwargs: dict | None):
        src = dict(kwargs or {})

        if accepts_gt:
            if "ground_truth" not in src and "target" in src:
                src["ground_truth"] = src["target"]
            src["ground_truth"] = _normalize_gt(src.get("ground_truth", None))

        if accepts_extra:
            extra_info = {}
            existing = src.get("extra_info", {})
            if isinstance(existing, dict):
                extra_info.update(existing)
            for k in list(src.keys()):
                if k not in ("ground_truth", "data_source", "extra_info"):
                    if k not in fn_params:
                        extra_info[k] = src.pop(k)
            src["extra_info"] = extra_info
        else:
            for k in list(src.keys()):
                if k not in fn_params:
                    src.pop(k, None)

        if solution_kw is not None:
            src[solution_kw] = pred_text  # pass model text under the right kw

        return {k: v for k, v in src.items() if k in fn_params}

    def reward_value(pred_text: str, kwargs: dict | None = None) -> float:
        out = fn(**_build_kwargs(pred_text, kwargs))
        return float(out["reward"]) if isinstance(out, dict) else float(out)

    def reward_with_info(pred_text: str, kwargs: dict | None = None):
        out = fn(**_build_kwargs(pred_text, kwargs))
        if isinstance(out, dict):
            return float(out.get("reward", 0.0)), dict(out.get("reward_info", {}))
        return float(out), {}

    return reward_value, reward_with_info

# (Adapter design per attached reference.)  # :contentReference[oaicite:0]{index=0}


def _load_accl_module():
    """Load the accelerated countdown ES module."""
    module_path = os.path.join(os.path.dirname(__file__), "es_fine-tuning_countdown_accl.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Accelerated countdown module not found at {module_path}")
    spec = importlib.util.spec_from_file_location("es_countdown_accl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import accelerated module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def parse_numbers_from_context(context: str | None) -> Optional[List[int]]:
    if not context:
        return None
    start_idx = context.find('[')
    end_idx = context.find(']', start_idx + 1)
    if start_idx == -1 or end_idx == -1:
        return None
    inside = context[start_idx + 1:end_idx]
    tokens = re.findall(r"-?\d+", inside)
    if not tokens:
        return None
    return [int(tok) for tok in tokens]


def build_task_records(dataset: List[tuple[str, Optional[str], Dict[str, Any]]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for context, target, kwargs in dataset:
        reward_kwargs = dict(kwargs or {})

        numbers = reward_kwargs.get("numbers")
        if numbers is None:
            inferred = parse_numbers_from_context(context)
            if inferred is not None:
                reward_kwargs["numbers"] = inferred
        target_val = reward_kwargs.get("target")
        if target_val is None and target is not None:
            if isinstance(target, (int, float)):
                target_val = int(target)
            elif isinstance(target, str) and target.strip().lstrip('-').isdigit():
                target_val = int(target)
            if target_val is not None:
                reward_kwargs["target"] = target_val

        records.append({
            "context": context,
            "target": reward_kwargs.get("target"),
            "numbers": reward_kwargs.get("numbers"),
            "reward_kwargs": reward_kwargs,
        })
    return records


def make_logger(verbose: bool):
    def _log(message: str):
        if verbose:
            print(message)

    return _log

def set_precision_dtype(precision: str):
    """
    Set the precision dtype for PyTorch tensors.
    Return the corresponding torch.dtype and vllm dtype string.
    """
    precision = precision.lower()
    if precision == "fp16" or precision == "float16":
        return torch.float16, "float16"
    elif precision == "bf16" or precision == "bfloat16":
        return torch.bfloat16, "bfloat16"
    elif precision == "fp32" or precision == "float32":
        return torch.float32, "float32"
    else:  
        raise ValueError(f"Unsupported precision: {precision}")


def launch_ray_engines(
    ray_mod,
    placement_group_fn,
    placement_strategy_cls,
    engine_cls,
    num_engines: int,
    model_path: str,
    dtype: str,
):
    pgs = [placement_group_fn([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
    ray_mod.get([pg.ready() for pg in pgs])
    strategies = [
        placement_strategy_cls(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]
    engines = [
        ray_mod.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(engine_cls).remote(
            model=model_path,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype=dtype,
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        for strategy in strategies
    ]
    return engines, pgs


def cleanup_ray(ray_mod, engines, placement_groups, remove_pg_fn):
    for engine in engines:
        try:
            ray_mod.kill(engine)
        except Exception:
            pass
    for pg in placement_groups:
        try:
            remove_pg_fn(pg)
        except Exception:
            pass
    ray_mod.shutdown()


def compute_rewards_from_outputs(outputs, dataset_records, reward_value_fn):
    rewards: List[float] = []
    for output, example in zip(outputs, dataset_records):
        response = output.outputs[0].text if getattr(output, "outputs", None) else ""
        kwargs = example.get("reward_kwargs") or {}
        rewards.append(float(reward_value_fn(response, kwargs)))
    return rewards


def compute_token_counts_from_outputs(outputs) -> List[int]:
    token_counts: List[int] = []
    for output in outputs:
        if getattr(output, "outputs", None):
            first_output = output.outputs[0]
            token_ids = getattr(first_output, "token_ids", None)
            if token_ids is None:
                token_ids = getattr(first_output, "token_id", None)  # fallback if singular
            try:
                token_counts.append(len(token_ids) if token_ids is not None else 0)
            except TypeError:
                token_counts.append(0)
        else:
            token_counts.append(0)
    return token_counts


def collect_dataset_rewards(
    ray_mod,
    engine,
    dataset_records,
    reward_value_fn,
    sampling_params_cls,
    max_tokens: int,
    seed: int,
):
    if not dataset_records:
        return [], []
    prompts = [example["context"] for example in dataset_records]

    sampling = sampling_params_cls(temperature=0.0, seed=seed, max_tokens=max_tokens)
    handle = engine.generate.remote(prompts, sampling, use_tqdm=False)
    outputs = ray_mod.get(handle)
    rewards = compute_rewards_from_outputs(outputs, dataset_records, reward_value_fn)
    token_counts = compute_token_counts_from_outputs(outputs)
    return rewards, token_counts


def evaluate_dataset_metrics(
    ray_mod,
    engine,
    dataset_records,
    reward_value_fn,
    sampling_params_cls,
    max_tokens: int,
    seed: int,
):
    if not dataset_records:
        return None, None, None
    rewards, token_counts = collect_dataset_rewards(
        ray_mod,
        engine,
        dataset_records,
        reward_value_fn,
        sampling_params_cls,
        max_tokens,
        seed,
    )
    if not rewards:
        return 0.0, 0.0, float(np.mean(token_counts)) if token_counts else 0.0
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    accuracy = float(np.mean(rewards_arr >= 1.0))
    mean_reward = float(np.mean(rewards_arr))
    mean_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    return accuracy, mean_reward, mean_tokens


def build_normalized_coefficients(
    avg_rewards: np.ndarray,
    baseline_mean_reward: float | None = None,
    method: str = "norm",
):
    rewards_arr = np.asarray(avg_rewards, dtype=np.float32)
    baseline = float(baseline_mean_reward or 0.0)
    normalized_methods = {"norm", "norm_relu", "light_advantage"}
    if rewards_arr.size == 0:
        return rewards_arr, {
            "coeff_mean": 0.0,
            "coeff_std": 0.0,
            "baseline_mean_reward": baseline,
            "normalized": method in normalized_methods,
            "method": method,
        }

    if method in normalized_methods:
        coeff_source = rewards_arr
        if method == "light_advantage" and baseline_mean_reward is not None:
            baseline_arr = np.asarray([baseline_mean_reward], dtype=rewards_arr.dtype)
            coeff_source = np.concatenate((coeff_source, baseline_arr))
        coeff_mean = float(coeff_source.mean()) if coeff_source.size else 0.0
        coeff_std = float(coeff_source.std()) if coeff_source.size else 0.0
        zscores = (rewards_arr - coeff_mean) / (coeff_std + 1e-8)
        vals = np.maximum(zscores, 0.0) if method == "norm_relu" else zscores
    elif method in ADVANTAGE_METHODS:
        coeff_source = rewards_arr - baseline
        coeff_mean = float(coeff_source.mean())
        coeff_std = float(coeff_source.std())
        if coeff_std < 1e-8:
            scaled_advantage = np.zeros_like(coeff_source)
        else:
            scaled_advantage = coeff_source / (coeff_std + 1e-8)
        if method == "advantage_base":
            vals = coeff_source
        elif method == "advantage_scaled":
            vals = scaled_advantage
        elif method == "advantage_scaled_relu":
            vals = np.maximum(scaled_advantage, 0.0)
        else:
            raise ValueError(f"Unknown advantage method: {method}")
    else:
        raise ValueError(f"Unsupported reward method: {method}")

    coeffs = (ALPHA / POPULATION_SIZE) * vals
    return coeffs, {
        "coeff_mean": coeff_mean,
        "coeff_std": coeff_std,
        "baseline_mean_reward": baseline,
        "normalized": method in normalized_methods,
        "method": method,
    }

import re
import torch

import torch

def convert_fused_qkv_and_gate_up_to_hf(
    state_dict: dict,
    base_model,
    verbose: bool = False,
) -> dict:
    """
    Convert vLLM-style fused qkv_proj / gate_up_proj weights into
    HuggingFace Qwen-style q_proj / k_proj / v_proj and
    gate_proj / up_proj.

    - Assumes:
        * attention linear:   qkv_proj  -> (Q, K, V) in that order
        * MLP linear:         gate_up_proj -> (gate, up) in that order
    - Uses base_model.config to infer head_dim, num_heads, num_kv_heads,
      and intermediate_size, and infers which weight dimension is fused
      by matching against (hidden_size, fused_dim).

    If shapes don't line up for some layer, that layer is skipped with a warning.
    """
    cfg = base_model.config
    hidden_size = cfg.hidden_size
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    num_layers = cfg.num_hidden_layers

    # Qwen-style FFN hidden dim
    intermediate_size = getattr(
        cfg, "intermediate_size",
        getattr(cfg, "ffn_hidden_size", None)
    )
    if intermediate_size is None:
        raise ValueError(
            "Could not find intermediate_size/ffn_hidden_size on config; "
            "needed for splitting gate_up_proj."
        )

    head_dim = hidden_size // num_heads
    q_size = num_heads * head_dim           # should equal hidden_size
    kv_size = num_kv_heads * head_dim       # K and V each
    fused_qkv_dim = q_size + 2 * kv_size    # dimension of fused QKV

    fused_gate_up_dim = 2 * intermediate_size

    new_state = dict(state_dict)  # shallow copy

    # Helper: figure out which axis of a weight is fused
    def _split_matrix(mat: torch.Tensor, fused_dim: int, sizes: tuple[int, ...], layer_idx: int, tag: str):
        """
        Split a 2D matrix into chunks along the axis whose size == fused_dim.
        Returns:
            axis, [chunk0, chunk1, ...] or (None, None) if mismatch.
        """
        if mat.ndim != 2:
            if verbose:
                print(f"[Converter][layer {layer_idx}][{tag}] expected 2D weight, got {mat.shape}")
            return None, None

        h0, h1 = mat.shape
        if h0 == fused_dim:
            axis = 0
        elif h1 == fused_dim:
            axis = 1
        else:
            if verbose:
                print(
                    f"[Converter][layer {layer_idx}][{tag}] no dimension equals fused_dim={fused_dim}; "
                    f"weight shape={mat.shape}"
                )
            return None, None

        if axis == 0:
            # mat: [fused_dim, other]
            chunks = []
            start = 0
            for sz in sizes:
                end = start + sz
                chunks.append(mat[start:end, :])
                start = end
        else:
            # mat: [other, fused_dim]
            chunks = []
            start = 0
            for sz in sizes:
                end = start + sz
                chunks.append(mat[:, start:end])
                start = end

        return axis, chunks

    # Helper: split 1D bias according to sizes
    def _split_bias(vec: torch.Tensor, fused_dim: int, sizes: tuple[int, ...], layer_idx: int, tag: str):
        if vec.ndim != 1:
            return None

        if vec.shape[0] != fused_dim:
            return None

        chunks = []
        start = 0
        for sz in sizes:
            end = start + sz
            chunks.append(vec[start:end])
            start = end
        return chunks

    # ---------------------------
    # Per-layer conversion
    # ---------------------------
    for layer_idx in range(num_layers):
        attn_prefix = f"model.layers.{layer_idx}.self_attn"
        mlp_prefix = f"model.layers.{layer_idx}.mlp"

        # ---- 1) QKV: qkv_proj.(weight|bias) -> q_proj, k_proj, v_proj ----
        qkv_w_key = f"{attn_prefix}.qkv_proj.weight"
        qkv_b_key = f"{attn_prefix}.qkv_proj.bias"

        if qkv_w_key in new_state:
            fused_W = new_state[qkv_w_key]

            axis, chunks = _split_matrix(
                fused_W,
                fused_qkv_dim,
                (q_size, kv_size, kv_size),
                layer_idx,
                "qkv_proj.weight",
            )
            if chunks is not None:
                q_W, k_W, v_W = chunks
                new_state[f"{attn_prefix}.q_proj.weight"] = q_W
                new_state[f"{attn_prefix}.k_proj.weight"] = k_W
                new_state[f"{attn_prefix}.v_proj.weight"] = v_W

                # Remove fused key so it doesn't show up as unexpected
                del new_state[qkv_w_key]
            else:
                if verbose:
                    print(f"[Converter][layer {layer_idx}] failed to split qkv_proj.weight; leaving fused key.")

        if qkv_b_key in new_state:
            fused_b = new_state[qkv_b_key]

            chunks = _split_bias(
                fused_b,
                fused_qkv_dim,
                (q_size, kv_size, kv_size),
                layer_idx,
                "qkv_proj.bias",
            )
            if chunks is not None:
                q_b, k_b, v_b = chunks
                new_state[f"{attn_prefix}.q_proj.bias"] = q_b
                new_state[f"{attn_prefix}.k_proj.bias"] = k_b
                new_state[f"{attn_prefix}.v_proj.bias"] = v_b

                del new_state[qkv_b_key]
            else:
                if verbose:
                    print(f"[Converter][layer {layer_idx}] failed to split qkv_proj.bias; leaving fused key.")

        # ---- 2) MLP: gate_up_proj.(weight|bias) -> gate_proj, up_proj ----
        gate_up_w_key = f"{mlp_prefix}.gate_up_proj.weight"
        gate_up_b_key = f"{mlp_prefix}.gate_up_proj.bias"

        if gate_up_w_key in new_state:
            fused_W = new_state[gate_up_w_key]

            axis, chunks = _split_matrix(
                fused_W,
                fused_gate_up_dim,
                (intermediate_size, intermediate_size),
                layer_idx,
                "gate_up_proj.weight",
            )
            if chunks is not None:
                gate_W, up_W = chunks
                new_state[f"{mlp_prefix}.gate_proj.weight"] = gate_W
                new_state[f"{mlp_prefix}.up_proj.weight"] = up_W

                del new_state[gate_up_w_key]
            else:
                if verbose:
                    print(f"[Converter][layer {layer_idx}] failed to split gate_up_proj.weight; leaving fused key.")

        if gate_up_b_key in new_state:
            fused_b = new_state[gate_up_b_key]
            
            chunks = _split_bias(
                fused_b,
                fused_gate_up_dim,
                (intermediate_size, intermediate_size),
                layer_idx,
                "gate_up_proj.bias",
            )
            if chunks is not None:
                gate_b, up_b = chunks
                new_state[f"{mlp_prefix}.gate_proj.bias"] = gate_b
                new_state[f"{mlp_prefix}.up_proj.bias"] = up_b

                del new_state[gate_up_b_key]
            else:
                if verbose:
                    print(f"[Converter][layer {layer_idx}] failed to split gate_up_proj.bias; leaving fused key.")

    if verbose:
        # Optional: quick summary of remaining fused keys (if any)
        leftover_fused = [
            k for k in new_state.keys()
            if ".qkv_proj." in k or ".gate_up_proj." in k
        ]
        if leftover_fused:
            print("[Converter] WARNING: leftover fused keys after conversion:")
            for k in leftover_fused:
                print("   ", k)

    return new_state



def save_remote_checkpoint(
    ray_mod,
    engine,
    iteration: int,
    tag: str,
    args,
    base_model_path: Path,
    train_size: int,
    tokenizer,
    torch_dtype,
):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_dir = Path(args.ckpt_dir) / (
        f"{args.model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{iteration}_"
        f"sigma{SIGMA}_alpha{ALPHA}_{args.precision}_procs{args.num_processes}_question_num{train_size}_{tag}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    tmp_weights = save_dir / "pytorch_model.vllm_fused.pth"

    # 1) Ask vLLM engine to dump its fused state_dict
    ray_mod.get(engine.collective_rpc.remote("save_self_weights_to_disk", args=(str(tmp_weights),)))

    # 2) Load base HF model
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch_dtype,
        device_map="cpu"
    )

    # 3) Load fused state dict and convert
    state_dict = torch.load(tmp_weights, map_location="cpu")

    # DEBUG: print the shapes of a couple of weights once
    if getattr(args, "verbose", False):
        w0 = state_dict.get("model.layers.0.self_attn.qkv_proj.weight", None)
        if w0 is not None:
            print("[Checkpoint DEBUG] qkv_proj.weight[0] shape:", w0.shape)
        g0 = state_dict.get("model.layers.0.mlp.gate_up_proj.weight", None)
        if g0 is not None:
            print("[Checkpoint DEBUG] gate_up_proj.weight[0] shape:", g0.shape)

    state_dict = convert_fused_qkv_and_gate_up_to_hf(state_dict, base_model, verbose=getattr(args, "verbose", False))


    # Fix for missing lm_head.weight if it is tied to embed_tokens
    # vLLM's save_self_weights_to_disk uses named_parameters() which dedups tied weights.
    if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    # 4) Load converted weights into the HF model
    missing, unexpected = base_model.load_state_dict(state_dict, strict=False)

    if getattr(args, "verbose", False):
        print(f"[Checkpoint] missing_keys (len={len(missing)}):")
        print(missing)
        print(f"[Checkpoint] unexpected_keys (len={len(unexpected)}):")
        print(unexpected)

    # 5) Save as a normal HF checkpoint
    base_model.tie_weights()
    base_model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    # 6) Cleanup
    del base_model
    gc.collect()
    tmp_weights.unlink(missing_ok=True)


def start_engine_eval(
    ray_mod,
    engine,
    eng_idx: int,
    seed_value: int,
    sigma: float,
    dataset_records,
    sampling_params_cls,
    max_tokens: int,
    inflight: Dict[Any, Dict[str, Any]],
    verbose: bool = False,
    log_fn = None,
    iteration_num: int = 0,
    seed_seq: int = 0,
    total_seeds: int = 0,
):
    ray_mod.get(engine.collective_rpc.remote("perturb_self_weights", args=(seed_value, sigma, False)))
    prompts = [example["context"] for example in dataset_records]

    sampling = sampling_params_cls(temperature=0.0, seed=seed_value, max_tokens=max_tokens)

    
    handle = engine.generate.remote(prompts, sampling, use_tqdm=False)
    inflight[handle] = {
        "engine": engine,
        "engine_idx": eng_idx,
        "seed": seed_value,
        "start_ts": time.time(),
        "iteration": iteration_num,
        "seq": seed_seq,
        "total": total_seeds,
    }
    if verbose and log_fn is not None:
        prefix = f"[Iter {iteration_num}][Engine {eng_idx}]"
        if total_seeds:
            log_fn(f"{prefix} scheduled seed {seed_seq}/{total_seeds} (id={seed_value})")
        else:
            log_fn(f"{prefix} scheduled seed id={seed_value}")


def register_signal_handlers(cleanup_fn):
    def _handler(sig, frame):
        cleanup_fn()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

# ----------------- CLI -----------------
parser = argparse.ArgumentParser()

ADVANTAGE_METHODS = ("advantage_base", "advantage_scaled", "advantage_scaled_relu")
REWARD_METHOD_CHOICES = ("norm", "norm_relu", "light_advantage", "advantage") + ADVANTAGE_METHODS
ADVANTAGE_ALIAS_MAP = {"advantage": "advantage_base"}

parser.add_argument('--task', type=str, choices=['countdown', 'gsm_8k', 'math', 'olympiadbench'], default='countdown',
                    help='Preset helper for dataset and reward defaults')
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='hf_cache')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--num_processes', type=int, default=4, help='Number of Ray/vLLM engines to launch (GPUs)')
parser.add_argument('--num_iterations', type=int, default=1000, help='Number of ES iterations to run')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
parser.add_argument('--data_sample', type=int, default=1000, help='Max training examples to use')
parser.add_argument('--reward_method', type=str, choices=REWARD_METHOD_CHOICES, default='norm',
                    help='Reward coefficient strategy: "norm"=z-score, "norm_relu"=ReLU(z-score), '
                         '"light_advantage"=z-score w/ baseline sample, '
                         '"advantage_base"=x-baseline, "advantage_scaled"=scaled advantage, '
                         '"advantage_scaled_relu"=ReLU(scaled advantage)')
parser.add_argument('--population_size', type=int, default=30, help='Population size per ES iteration')

# Data sources (either JSON fallback or Parquet)
parser.add_argument('--train_json', type=str, default=None, help='JSON [{context,target}] (fallback if no parquet)')
parser.add_argument('--val_json', type=str, default=None, help='JSON [{context,target}] for validation')
parser.add_argument('--train_parquet', type=str, default=None, help='Path to training parquet split')
parser.add_argument('--val_parquet', type=str, default=None, help='Path to validation parquet split')

# Parquet slicing / limits (similar to attached)
parser.add_argument('--train_slice', type=str, default=None, help='Slice string like "0:1000" or "1000:"')
parser.add_argument('--train_offset', type=int, default=0, help='Offset if no slice is provided')
parser.add_argument('--train_limit', type=int, default=None, help='Max rows after offset if no slice is provided')
parser.add_argument('--val_slice', type=str, default=None, help='Slice string for val')
parser.add_argument('--val_offset', type=int, default=0, help='Val offset if no slice is provided')
parser.add_argument('--val_limit', type=int, default=None, help='Val max rows after offset')
parser.add_argument('--context_char_limit', type=int, default=None, help='Truncate contexts to this many chars')

# Reward adapter
parser.add_argument("--reward_fn_file", type=str, default=None,
                    help="Path to Python file exporting reward_function(...)")
parser.add_argument("--reward_fn_name", type=str, default=None,
                    help="Function name in reward file")
parser.add_argument("--reward_end_token", default=None, type=str,
                    help="Optional end token to pass via reward kwargs (no generation stopping)")

# Eval cadence
parser.add_argument('--eval_every', type=int, default=1, help='Evaluate train/val accuracy every N iters (0=end only)')
parser.add_argument('--eval_batch_size', type=int, default=32,
                    help='Batch size for generation during evaluation')
                    
# Checkpoint cadence
parser.add_argument('--save_model_every', type=int, default=50, help='Save checkpoint every N iterations (0=never)')
parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Checkpoint base directory')

#meta data
parser.add_argument('--logs_dir', type=str, default='logs', help='Directory for meta and CSV metrics')
parser.add_argument('--save_rewards_per_iter', action='store_true', help='Save per-iteration reward vectors (.npy)')

# W&B
parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity')
parser.add_argument('--wandb_project', type=str, default='es-countdown', help='W&B project')
parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')

#Chat template!
parser.add_argument('--apply_chat_template', action='store_true', help='Apply chat template or not')
parser.add_argument('--chat_template_file', type=str, default=None, help='Filepath to the chat template')


args = parser.parse_args()
if args.reward_method in ADVANTAGE_ALIAS_MAP:
    args.reward_method = ADVANTAGE_ALIAS_MAP[args.reward_method]

_SCRIPT_DIR = Path(__file__).resolve().parent
_PRESET_DEFAULTS = {
    "countdown": {
        "train_json": _SCRIPT_DIR / "countdown" / "data" / "countdown.json",
        "reward_fn_file": _SCRIPT_DIR / "countdown_reward_micah.py",
        "reward_fn_name": "reward_function",
    },
    "gsm_8k": {
        "train_parquet": _SCRIPT_DIR / "gsm_8k" / "train.parquet",
        "val_parquet": _SCRIPT_DIR / "gsm_8k" / "test.parquet",
        "reward_fn_file": _SCRIPT_DIR / "gsm_8k_reward.py",
        "reward_fn_name": "my_reward_fn",
    },
    "math": {
        "train_parquet": _SCRIPT_DIR / "math" / "train.parquet",
        "val_parquet": _SCRIPT_DIR / "math" / "test.parquet",
        "reward_fn_file": _SCRIPT_DIR / "gsm_8k_reward.py",
        "reward_fn_name": "my_reward_fn",
    },
    "olympiadbench": {
        "train_parquet": _SCRIPT_DIR / "olympiadbench" / "train.parquet",
        "val_parquet": _SCRIPT_DIR / "olympiadbench" / "test.parquet",
        "reward_fn_file": _SCRIPT_DIR / "gsm_8k_reward.py",
        "reward_fn_name": "my_reward_fn",
    }
}

preset_defaults = _PRESET_DEFAULTS.get(args.task, {})

if args.task == "countdown":
    if not args.train_parquet and not args.train_json:
        default_train_json = preset_defaults.get("train_json")
        if isinstance(default_train_json, Path) and default_train_json.exists():
            args.train_json = str(default_train_json)
elif args.task in {"gsm_8k", "math", "olympiadbench"}:
    default_train_parquet = preset_defaults.get("train_parquet")
    if not args.train_parquet and isinstance(default_train_parquet, Path) and default_train_parquet.exists():
        args.train_parquet = str(default_train_parquet)
    default_val_parquet = preset_defaults.get("val_parquet")
    if not args.val_parquet and isinstance(default_val_parquet, Path) and default_val_parquet.exists():
        args.val_parquet = str(default_val_parquet)

if not args.reward_fn_file:
    default_reward_file = preset_defaults.get("reward_fn_file")
    if isinstance(default_reward_file, Path) and default_reward_file.exists():
        args.reward_fn_file = str(default_reward_file)

if not args.reward_fn_file:
    parser.error("--reward_fn_file is required unless provided by a known task preset")

if not args.reward_fn_name:
    args.reward_fn_name = preset_defaults.get("reward_fn_name", "reward_function")

# ----------------- ES Hyperparameters (unchanged) -----------------
NUM_ITERATIONS = 1000                   # number of ES iterations
POPULATION_SIZE = args.population_size  # perturbations per iteration
SIGMA = 0.001                           # noise scale
ALPHA = 0.0005                          # learning rate
max_new_tokens = 1024                   # decoding cap
do_sample = False                       # greedy decoding
initial_seed = 33                       # RNG seed for ES

# ----------------- Simple helpers -----------------
def force_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def save_model_checkpoint(model, tokenizer, iteration, model_name, initial_seed, args, dataset_size):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    question_num = dataset_size
    save_dir = os.path.join(
        args.ckpt_dir,
        f"{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{iteration}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_procs{args.num_processes}_question_num{question_num}_checkpoint"
    )
    print(f"Saving checkpoint at iteration {iteration} to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Checkpoint saved successfully.")

# ----------------- Parquet helpers (from attached ref) -----------------
def _concat_prompt(prompt_obj) -> str:
    """
    Parquet 'prompt' may be a list of {role, content} chunks; join safely.
    """
    p = prompt_obj
    if isinstance(p, list) and p:
        parts = []
        for m in p:
            if isinstance(m, dict):
                parts.append(f"{m.get('role','user')}: {m.get('content','')}")
            else:
                parts.append(str(m))
        return "\n".join(parts)
    if isinstance(p, dict):
        return p.get("content", "") or str(p)
    return str(p) if p is not None else ""
# :contentReference[oaicite:1]{index=1}

def _parse_slice_str(slice_str: str) -> slice:
    parts = slice_str.split(':')
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid slice string: {slice_str}")
    parts = [(int(x) if x != '' else None) for x in parts]
    while len(parts) < 3:
        parts.append(None)
    start, stop, step = parts
    return slice(start, stop, step)
# :contentReference[oaicite:2]{index=2}

def load_parquet_split(path: str,
                       limit: int | None = None,
                       slice_str: str | None = None,
                       offset: int = 0,
                       hard_limit: int | None = None,
                       context_char_limit: int | None = None,
                       end_token: str | None = None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)

    # slicing behavior
    if slice_str:
        s = _parse_slice_str(slice_str)
        df = df.iloc[s]
    else:
        if offset or hard_limit:
            stop = None if hard_limit is None else offset + max(int(hard_limit), 0)
            df = df.iloc[offset:stop]
    if limit is not None:
        df = df.head(limit)

    contexts, golds, reward_kwargs = [], [], []

    for _, row_s in df.iterrows():
        row = row_s.to_dict()

        prompt_obj = row.get("prompt")
        if prompt_obj is None:
            prompt_obj = row.get("context")
        
        
        if prompt_obj and isinstance(prompt_obj[0], dict):
            if args.apply_chat_template:
                #print("we're only applying chat template to the content!")
                ctx = prompt_obj[0].get("content", "")
            else:
                #print("We passed but args.apply_chat_template is off")
                ctx = _concat_prompt(prompt_obj)
        else:
            #print("We skipped everything")
            ctx = _concat_prompt(prompt_obj)

        if context_char_limit is not None and isinstance(ctx, str):
            ctx = ctx[:context_char_limit]

        # keep gold if present (optional)
        gt = None
        rm = row.get("reward_model")
        if isinstance(rm, dict) and rm.get("ground_truth") is not None:
            gt = str(rm["ground_truth"])
        if gt is None and row.get("solution") is not None:
            gt = str(row["solution"])

        # per-example kwargs for reward function (
        # support legacy parquet schema where values live under reward_model.extra_info
        # as well as the new schema with top-level columns).
        extra = row.get("extra_info") or {}
        nums = extra.get("numbers", None)
        if nums is None and row.get("numbers") is not None:
            nums = row["numbers"]
        if isinstance(nums, np.ndarray):
            nums = nums.tolist()
        if isinstance(nums, (list, tuple)):
            nums = [int(x) for x in nums]

        tgt = extra.get("target", None)
        if tgt is None and row.get("target") is not None:
            tgt = row["target"]
        try:
            tgt = int(tgt) if tgt is not None else None
        except Exception:
            tgt = None

        if ctx:
            contexts.append(ctx)
            golds.append(gt)
            reward_kwargs.append({
                "numbers": nums,
                "target": tgt,
                "end_token": end_token,  # just passthrough; NOT used to stop generation
            })

    return contexts, golds, reward_kwargs
# (Function mirrors attached behavior.)  # :contentReference[oaicite:3]{index=3}


def read_text_or_none(p: Path) -> str | None:
    try:
        s = p.read_text(encoding="utf-8")
        return s.strip() if s else None
    except Exception:
        return None

def find_chat_template(explicit_file: Optional[str],
                       tokenizer,
                       consolidated_dir: Path,
                       raw_actor_dir: Path) -> str | None:
    # 0) explicit flag or env
    if explicit_file:
        s = read_text_or_none(Path(explicit_file))
        if s: return s
    env = os.environ.get("CHAT_TEMPLATE_OVERRIDE")
    if env:
        s = read_text_or_none(Path(env))
        if s: return s

    # 1) tokenizer already has one
    t = getattr(tokenizer, "chat_template", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    # 2) file in consolidated dir
    s = read_text_or_none(consolidated_dir / "chat_template.jinja")
    if s: return s

    # 3) file in original raw dir (pre-consolidation)
    s = read_text_or_none(raw_actor_dir / "chat_template.jinja")
    if s: return s
    # Return error if nothing found
    raise ValueError(
        "No chat template found. "
        "Either add chat_template.jinja next to the checkpoint, "
        "or ensure tokenizer_config.json contains `chat_template`, "
        "or pass --chat_template_file."
    )

def build_prompts(records,
                  tokenizer,
                  apply_chat_template: bool,
                  chat_template_text: Optional[str]) -> list[str]:
    if not apply_chat_template:
        return [r["context"] for r in records]

    if not (isinstance(chat_template_text, str) and chat_template_text.strip()):
        raise ValueError(
            "apply_chat_template=True but no chat template found. "
            "Either add chat_template.jinja next to the checkpoint, "
            "or ensure tokenizer_config.json contains `chat_template`, "
            "or pass --chat_template_file."
        )

    # set it on the tokenizer for downstream libs
    try:
        tokenizer.chat_template = chat_template_text
    except Exception:
        pass

    outs = []
    for r in records:
        msgs = [{"role": "user", "content": r["context"]}]
        s = tokenizer.apply_chat_template(
            msgs,
            chat_template=chat_template_text,
            tokenize=False,
            add_generation_prompt=True,
        )
        outs.append(s)
    return outs
# ----------------- Main ES loop -----------------
def main():
    import pathlib
    import random
    import shutil

    global NUM_ITERATIONS
    NUM_ITERATIONS = max(1, args.num_iterations)

    accl_module = _load_accl_module()
    ray = accl_module.ray
    placement_group = accl_module.placement_group
    remove_placement_group = accl_module.remove_placement_group
    PlacementGroupSchedulingStrategy = accl_module.PlacementGroupSchedulingStrategy
    SamplingParams = accl_module.SamplingParams
    ESNcclLLM = accl_module.ESNcclLLM
    get_ip = accl_module.get_ip
    get_open_port = accl_module.get_open_port
    vprint = make_logger(args.verbose)
    advantage_mode = args.reward_method if args.reward_method in ADVANTAGE_METHODS else None

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = pathlib.Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{run_ts}_model={args.model_name.replace('/','_')}_seed={initial_seed}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}"
    metrics_csv = logs_dir / f"metrics_{run_id}.csv"
    meta_json = logs_dir / f"meta_{run_id}.json"

    if torch.cuda.is_available():
        print("Surely not, we're using GPU")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory
    else:
        print("Are we using cpu?")
        gpu_name, total_mem = "cpu", 0

    meta = {
        "timestamp": run_ts,
        "model_name": args.model_name,
        "precision": args.precision,
        "population_size": POPULATION_SIZE,
        "num_iterations": NUM_ITERATIONS,
        "sigma": SIGMA,
        "alpha": ALPHA,
        "eval_batch_size": args.eval_batch_size,
        "num_processes": args.num_processes,
        "data_sample": args.data_sample,
        "train_parquet": args.train_parquet,
        "val_parquet": args.val_parquet,
        "train_json": args.train_json,
        "val_json": args.val_json,
        "eval_every": args.eval_every,
        "save_model_every": args.save_model_every,
        "world_size": max(1, args.num_processes),
        "rank": 0,
        "device": "ray",
        "task": args.task,
        "gpu_name": gpu_name,
        "gpu_total_mem_bytes": int(total_mem),
        "reward_fn_file": args.reward_fn_file,
        "reward_fn_name": args.reward_fn_name,
        "reward_end_token": args.reward_end_token,
        "logs_csv": str(metrics_csv),
        "reward_method": args.reward_method,
    }
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    if not metrics_csv.exists():
        with open(metrics_csv, "w") as f:
            f.write(
                "iteration,mean_reward,min_reward,max_reward,baseline_mean_reward,train_acc,val_acc,"
                "train_mean_reward_eval,val_mean_reward_eval,train_mean_tokens_iter,baseline_mean_tokens,"
                "train_mean_tokens_eval,val_mean_tokens_eval,iter_time_sec\n"
            )

    reward_value_fn, _ = make_reward_adapters(args.reward_fn_file, args.reward_fn_name)

    train_dataset = []
    if args.train_parquet:
        tr_ctx, tr_gold, tr_kwargs = load_parquet_split(
            args.train_parquet,
            limit=args.train_limit,
            slice_str=args.train_slice,
            offset=args.train_offset,
            hard_limit=args.train_limit,
            context_char_limit=args.context_char_limit,
            end_token=args.reward_end_token
        )
        train_dataset = [(c, (g if g is not None else ""), rk) for c, g, rk in zip(tr_ctx, tr_gold, tr_kwargs)]
    else:
        default_path = args.train_json or os.path.join(os.path.dirname(__file__), 'data/countdown.json')
        if not os.path.exists(default_path):
            raise FileNotFoundError(f"Dataset file not found: {default_path}")
        with open(default_path, 'r') as f:
            data_json = json.load(f)
        for item in data_json:
            context = item['context']
            target = item['target']
            train_dataset.append((context, target, {"end_token": args.reward_end_token}))

    val_dataset = train_dataset[args.data_sample:min(args.data_sample + 500, len(train_dataset))] if args.data_sample else []
    train_dataset = train_dataset[:args.data_sample] if args.data_sample else train_dataset

    
    if args.val_parquet:
        v_ctx, v_gold, v_kwargs = load_parquet_split(
            args.val_parquet,
            limit=args.val_limit,
            slice_str=args.val_slice,
            offset=args.val_offset,
            hard_limit=args.val_limit,
            context_char_limit=args.context_char_limit,
            end_token=args.reward_end_token
        )
        val_dataset = [(c, (g if g is not None else ""), rk) for c, g, rk in zip(v_ctx, v_gold, v_kwargs)]
    elif args.val_json:
        if not os.path.exists(args.val_json):
            raise FileNotFoundError(f"Validation file not found: {args.val_json}")
        with open(args.val_json, 'r') as f:
            val_json = json.load(f)
        for item in val_json:
            context = item['context']
            target = item['target']
            val_dataset.append((context, target, {"end_token": args.reward_end_token}))

    vprint(f"Preparing {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    print(f"Task preset: {args.task}")
    print(f"Loaded {len(train_dataset)} training samples")
    if val_dataset:
        print(f"Loaded {len(val_dataset)} validation samples")
    print(f"Num processes (engines): {args.num_processes}")
    print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
    print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")

    train_records = build_task_records(train_dataset)
    val_records = build_task_records(val_dataset)

    
    precision_key = args.precision.lower()
    torch_dtype, vllm_dtype = set_precision_dtype(precision_key)

    print(f"Loading base model {args.model_name} with dtype {torch_dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        cache_dir=args.hf_cache_dir,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, cache_dir=args.hf_cache_dir)
    
    #To debug and print first raw prompt
    if args.verbose and train_records:
        sample_prompt = train_records[0]["context"]
        sample_ids = tokenizer(sample_prompt, return_tensors="pt").input_ids[0]
        decoded = tokenizer.decode(sample_ids, skip_special_tokens=False)
        print("Decoded example from training dataset:", decoded)

    if args.apply_chat_template:
        chat_tmpt = find_chat_template(
            explicit_file=args.chat_template_file,
            tokenizer=tokenizer,
            consolidated_dir=None,
            raw_actor_dir=None,
        )

        train_templated = build_prompts(
            records=train_records,
            tokenizer=tokenizer,
            apply_chat_template=True,
            chat_template_text=chat_tmpt,
        )
        val_templated = build_prompts(
            records=val_records,
            tokenizer=tokenizer,
            apply_chat_template=True,
            chat_template_text=chat_tmpt,
        )

        for rec, prompt in zip(train_records, train_templated):
            rec["context"] = prompt
        for rec, prompt in zip(val_records, val_templated):
            rec["context"] = prompt
    
    print(f"TRAINING DATASET SIZE: {len(train_records)}")
    print(f"Validation dataset size: {len(val_records)}")

    #To debug and print first raw prompt
    if args.verbose and train_records:
        sample_prompt = train_records[0]["context"]
        sample_ids = tokenizer(sample_prompt, return_tensors="pt").input_ids[0]
        decoded = tokenizer.decode(sample_ids, skip_special_tokens=False)
        print("Decoded example from training dataset:", decoded)
    
    run_dir = logs_dir / f"{run_id}_accl"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_saves_dir = run_dir / "model_saves"
    model_saves_dir.mkdir(parents=True, exist_ok=True)

    base_model_path = model_saves_dir / "base_model"
    if base_model_path.exists():
        shutil.rmtree(base_model_path)
    base_model_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(base_model_path))
    base_model.save_pretrained(str(base_model_path))
    del base_model
    force_memory_cleanup()

    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    num_engines = max(1, args.num_processes)
    engines, placement_groups = launch_ray_engines(
        ray,
        placement_group,
        PlacementGroupSchedulingStrategy,
        ESNcclLLM,
        num_engines,
        str(base_model_path),
        vllm_dtype,
    )

    master_address = get_ip()
    master_port = get_open_port()
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, num_engines)
        )
        for i in range(num_engines)
    ])
    vprint("Initialized inter-engine NCCL group")

    cleanup_fn = functools.partial(
        cleanup_ray,
        ray,
        engines,
        placement_groups,
        remove_placement_group,
    )
    register_signal_handlers(cleanup_fn)

    rewards_dir = run_dir / "rewards"
    rewards_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb

            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "model_name": args.model_name,
                    "precision": args.precision,
                    "population_size": POPULATION_SIZE,
                    "num_iterations": NUM_ITERATIONS,
                    "sigma": SIGMA,
                    "alpha": ALPHA,
                    "train_size": len(train_records),
                    "val_size": len(val_records),
                    "reward_method": args.reward_method,
                },
            )
            wandb.define_metric("iteration")
        except Exception as exc:
            print(f"[W&B] init failed: {exc}")
            use_wandb = False

    if args.verbose and train_records:
        sample_prompt = train_records[0]["context"]
        sample_ids = tokenizer(sample_prompt, return_tensors="pt").input_ids[0]
        decoded = tokenizer.decode(sample_ids, skip_special_tokens=False)
        vprint(f"Decoded example from training dataset:{decoded}" )

    
    vprint("Running initial evaluation (iteration 0)")
    train_acc0, train_mean0, train_tokens0 = (
        evaluate_dataset_metrics(
            ray,
            engines[0],
            train_records,
            reward_value_fn,
            SamplingParams,
            max_new_tokens,
            initial_seed,
        )
        if train_records
        else (None, None, None)
    )
    val_acc0, val_mean0, val_tokens0 = (
        evaluate_dataset_metrics(
            ray,
            engines[0],
            val_records,
            reward_value_fn,
            SamplingParams,
            max_new_tokens,
            initial_seed + 1,
        )
        if val_records
        else (None, None, None)
    )

    with open(metrics_csv, "a") as f:
        line_parts = [
            "0",
            "",
            "",
            "",
            "" if train_mean0 is None else f"{train_mean0:.6f}",
            "" if train_acc0 is None else f"{train_acc0:.6f}",
            "" if val_acc0 is None else f"{val_acc0:.6f}",
            "" if train_mean0 is None else f"{train_mean0:.6f}",
            "" if val_mean0 is None else f"{val_mean0:.6f}",
            "",  # no iteration token stat for iteration 0
            "" if train_tokens0 is None else f"{train_tokens0:.6f}",
            "" if train_tokens0 is None else f"{train_tokens0:.6f}",
            "" if val_tokens0 is None else f"{val_tokens0:.6f}",
            "0.0",
        ]
        f.write(",".join(line_parts) + "\n")

    if use_wandb and train_acc0 is not None and train_mean0 is not None:
        import wandb

        payload = {
            "iteration": 0,
            "train_acc": float(train_acc0),
            "train_mean_reward": float(train_mean0),
        }
        payload["baseline_mean_reward"] = float(train_mean0)
        if val_acc0 is not None and val_mean0 is not None:
            payload.update({
                "val_acc": float(val_acc0),
                "val_mean_reward": float(val_mean0),
            })
        if train_tokens0 is not None:
            payload["train_mean_tokens_eval"] = float(train_tokens0)
            payload["baseline_mean_tokens"] = float(train_tokens0)
        if val_tokens0 is not None:
            payload["val_mean_tokens_eval"] = float(val_tokens0)
        wandb.log(payload)
    if args.verbose and train_acc0 is not None and train_mean0 is not None:
        msg = f"[Eval 0] train_acc={train_acc0:.4f}, train_mean_reward={train_mean0:.4f}"
        if train_tokens0 is not None:
            msg += f", train_mean_tokens={train_tokens0:.2f}"
        if val_acc0 is not None and val_mean0 is not None:
            msg += f", val_acc={val_acc0:.4f}, val_mean_reward={val_mean0:.4f}"
            if val_tokens0 is not None:
                msg += f", val_mean_tokens={val_tokens0:.2f}"
        vprint(msg)

    np.random.seed(initial_seed)
    random.seed(initial_seed)
    total_start = time.time()

    try:
        for iteration in range(NUM_ITERATIONS):
            iter_start = time.time()
            baseline_rewards = []
            baseline_tokens = []
            baseline_mean_reward = 0.0
            baseline_mean_tokens = 0.0

            if train_records:
                baseline_seed = initial_seed + iteration
                baseline_rewards, baseline_tokens = collect_dataset_rewards(
                    ray,
                    engines[0],
                    train_records,
                    reward_value_fn,
                    SamplingParams,
                    max_new_tokens,
                    baseline_seed,
                )
                if baseline_rewards:
                    baseline_mean_reward = float(np.mean(baseline_rewards))
                if baseline_tokens:
                    baseline_mean_tokens = float(np.mean(baseline_tokens))
                if args.verbose:
                    vprint(
                        f"[Iter {iteration + 1}] baseline_mean_reward={baseline_mean_reward:.4f}, "
                        f"baseline_mean_tokens={baseline_mean_tokens:.2f}"
                    )
            seeds = [random.randint(0, 1_000_000) for _ in range(POPULATION_SIZE)]
            seeds_perf = {}
            inflight = {}

            total_seeds = len(seeds)
            seeds_with_idx = list(enumerate(seeds, start=1))
            seed_iter = iter(seeds_with_idx)

            for eng_idx, engine in enumerate(engines):
                try:
                    seq, seed = next(seed_iter)
                except StopIteration:
                    break
                start_engine_eval(
                    ray,
                    engine,
                    eng_idx,
                    seed,
                    SIGMA,
                    train_records,
                    SamplingParams,
                    max_new_tokens,
                    inflight,
                    args.verbose,
                    vprint,
                    iteration + 1,
                    seq,
                    total_seeds,
                )
            while inflight:
                done, _ = ray.wait(list(inflight.keys()), num_returns=1)
                handle = done[0]
                meta = inflight.pop(handle)
                outputs = ray.get(handle)
                elapsed = time.time() - meta["start_ts"]
                rewards = compute_rewards_from_outputs(outputs, train_records, reward_value_fn)
                token_counts = compute_token_counts_from_outputs(outputs)
                avg_reward = float(np.mean(rewards)) if rewards else 0.0
                avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0
                seeds_perf[meta["seed"]] = {
                    "avg_reward": avg_reward,
                    "rewards": rewards,
                    "elapsed": elapsed,
                    "avg_tokens": avg_tokens,
                    "token_counts": token_counts,
                }
                if args.verbose:
                    vprint(
                        f"[Iter {meta['iteration']}][Engine {meta['engine_idx']}] "
                        f"completed {meta['seq']}/{meta['total']} in {elapsed:.2f}s "
                        f"avg_reward={avg_reward:.4f}, mean_tokens={avg_tokens:.2f}"
                    )
                ray.get(meta["engine"].collective_rpc.remote("restore_self_weights", args=(meta["seed"], SIGMA)))
                try:
                    seq_next, seed_next = next(seed_iter)
                except StopIteration:
                    continue
                start_engine_eval(
                    ray,
                    meta["engine"],
                    meta["engine_idx"],
                    seed_next,
                    SIGMA,
                    train_records,
                    SamplingParams,
                    max_new_tokens,
                    inflight,
                    args.verbose,
                    vprint,
                    iteration + 1,
                    seq_next,
                    total_seeds,
                )
            
            if len(seeds_perf) != len(seeds):
                print("[Warning] Some seeds failed to evaluate; skipping iteration update")
                continue

            avg_rewards = np.asarray([seeds_perf[s]["avg_reward"] for s in seeds], dtype=np.float32)
            avg_tokens_arr = np.asarray([seeds_perf[s]["avg_tokens"] for s in seeds], dtype=np.float32)
            mean_reward = float(avg_rewards.mean()) if avg_rewards.size else 0.0
            std_reward = float(avg_rewards.std()) if avg_rewards.size else 0.0
            min_reward = float(avg_rewards.min()) if avg_rewards.size else 0.0
            max_reward = float(avg_rewards.max()) if avg_rewards.size else 0.0
            mean_tokens_iter = float(avg_tokens_arr.mean()) if avg_tokens_arr.size else 0.0
            if args.verbose:
                vprint(
                    f"[Iter {iteration + 1}] reward stats mean={mean_reward:.4f}, std={std_reward:.4f}, "
                    f"min={min_reward:.4f}, max={max_reward:.4f}, mean_tokens={mean_tokens_iter:.2f}"
                )

            coeffs, coeff_stats = build_normalized_coefficients(
                avg_rewards,
                baseline_mean_reward if train_records else None,
                args.reward_method,
            )

            update_handles = [
                engines[0].collective_rpc.remote("perturb_self_weights", args=(int(seed), float(coeff), False))
                for seed, coeff in zip(seeds, coeffs)
            ]
            ray.get(update_handles)
            ray.get([engine.collective_rpc.remote("broadcast_all_weights", args=(0,)) for engine in engines])
            if args.verbose:
                vprint(f"[Iter {iteration + 1}] applied ES update and broadcasted weights")

            iter_time = time.time() - iter_start

            if args.save_rewards_per_iter:
                np.save(str(rewards_dir / f"rewards_iter_{iteration + 1:04d}.npy"), avg_rewards)

            train_acc_eval = train_mean_eval = train_tokens_eval = None
            val_acc_eval = val_mean_eval = val_tokens_eval = None
            if args.eval_every == 0:
                should_eval = iteration == NUM_ITERATIONS - 1
            else:
                should_eval = (iteration + 1) % args.eval_every == 0
            if should_eval:
                train_acc_eval, train_mean_eval, train_tokens_eval = evaluate_dataset_metrics(
                    ray,
                    engines[0],
                    train_records,
                    reward_value_fn,
                    SamplingParams,
                    max_new_tokens,
                    initial_seed + iteration + 1,
                )
                if val_records:
                    val_acc_eval, val_mean_eval, val_tokens_eval = evaluate_dataset_metrics(
                        ray,
                        engines[0],
                        val_records,
                        reward_value_fn,
                        SamplingParams,
                        max_new_tokens,
                        initial_seed + iteration + 2,
                    )
                if args.verbose:
                    train_acc_txt = f"{train_acc_eval:.4f}" if train_acc_eval is not None else "n/a"
                    train_mean_txt = f"{train_mean_eval:.4f}" if train_mean_eval is not None else "n/a"
                    train_token_txt = f"{train_tokens_eval:.2f}" if train_tokens_eval is not None else "n/a"
                    if val_acc_eval is not None and val_mean_eval is not None:
                        if val_tokens_eval is not None:
                            val_txt = (
                                f", val_acc={val_acc_eval:.4f}, val_mean_reward={val_mean_eval:.4f}, "
                                f"val_mean_tokens={val_tokens_eval:.2f}"
                            )
                        else:
                            val_txt = f", val_acc={val_acc_eval:.4f}, val_mean_reward={val_mean_eval:.4f}"
                    else:
                        val_txt = ""
                    vprint(
                        f"[Eval {iteration + 1}] train_acc={train_acc_txt}, train_mean_reward={train_mean_txt}, "
                        f"train_mean_tokens={train_token_txt}{val_txt}"
                    )

            with open(metrics_csv, "a") as f:
                line_parts = [
                    f"{iteration + 1}",
                    f"{mean_reward:.6f}",
                    f"{min_reward:.6f}",
                    f"{max_reward:.6f}",
                    f"{baseline_mean_reward:.6f}",
                    "" if train_acc_eval is None else f"{train_acc_eval:.6f}",
                    "" if val_acc_eval is None else f"{val_acc_eval:.6f}",
                    "" if train_mean_eval is None else f"{train_mean_eval:.6f}",
                    "" if val_mean_eval is None else f"{val_mean_eval:.6f}",
                    f"{mean_tokens_iter:.6f}",
                    f"{baseline_mean_tokens:.6f}",
                    "" if train_tokens_eval is None else f"{train_tokens_eval:.6f}",
                    "" if val_tokens_eval is None else f"{val_tokens_eval:.6f}",
                    f"{iter_time:.6f}",
                ]
                f.write(",".join(line_parts) + "\n")

            if use_wandb:
                try:
                    import wandb

                    payload = {
                        "iteration": iteration + 1,
                        "mean_reward": mean_reward,
                        "min_reward": min_reward,
                        "max_reward": max_reward,
                        "iteration_time": iter_time,
                        "train_mean_tokens_iter": mean_tokens_iter,
                        "baseline_mean_reward": baseline_mean_reward,
                        "baseline_mean_tokens": baseline_mean_tokens,
                    }
                    if advantage_mode:
                        payload["coeff_source_mean"] = coeff_stats.get("coeff_mean", 0.0)
                        payload["coeff_source_std"] = coeff_stats.get("coeff_std", 0.0)
                    if should_eval:
                        if train_acc_eval is not None:
                            payload["train_acc"] = train_acc_eval
                        if train_mean_eval is not None:
                            payload["train_mean_reward_eval"] = train_mean_eval
                        if val_acc_eval is not None:
                            payload["val_acc"] = val_acc_eval
                        if val_mean_eval is not None:
                            payload["val_mean_reward_eval"] = val_mean_eval
                        if train_tokens_eval is not None:
                            payload["train_mean_tokens_eval"] = train_tokens_eval
                        if val_tokens_eval is not None:
                            payload["val_mean_tokens_eval"] = val_tokens_eval
                    wandb.log(payload)
                except Exception as exc:
                    if args.verbose:
                        print(f"[W&B] log failed: {exc}")

            if args.verbose:
                extra = ""
                if advantage_mode:
                    extra = (
                        f" coeff_mode={advantage_mode}"
                        f" coeff_src_mean={coeff_stats.get('coeff_mean', 0.0):.4f}"
                        f" coeff_src_std={coeff_stats.get('coeff_std', 0.0):.4f}"
                    )
                print(
                    f"Iteration {iteration + 1}/{NUM_ITERATIONS} | time={iter_time:.2f}s | "
                    f"mean={mean_reward:.4f} std={std_reward:.4f} min={min_reward:.4f} max={max_reward:.4f} "
                    f"mean_tokens={mean_tokens_iter:.2f} baseline_mean_reward={baseline_mean_reward:.4f} "
                    f"baseline_tokens={baseline_mean_tokens:.2f}{extra}"
                )

            if args.save_model_every and (iteration + 1) % args.save_model_every == 0:
                save_remote_checkpoint(
                    ray,
                    engines[0],
                    iteration + 1,
                    "checkpoint",
                    args,
                    base_model_path,
                    len(train_records),
                    tokenizer,
                    torch_dtype,
                )

        total_time = time.time() - total_start
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        save_remote_checkpoint(
            ray,
            engines[0],
            NUM_ITERATIONS,
            "final",
            args,
            base_model_path,
            len(train_records),
            tokenizer,
            torch_dtype,
        )
    finally:
        if use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass
        cleanup_fn()


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()
