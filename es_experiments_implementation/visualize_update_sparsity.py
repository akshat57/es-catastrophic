import argparse, re, math, json, os
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm

try:
    from safetensors.torch import load_file as safe_load
except Exception:
    safe_load = None

# ------------- helpers -------------
PATTERNS = {
    "Q":   re.compile(r"(?:\bq_proj\b|\.q\.)"),
    "K":   re.compile(r"(?:\bk_proj\b|\.k\.)"),
    "V":   re.compile(r"(?:\bv_proj\b|\.v\.)"),
    "O":   re.compile(r"(?:\bo_proj\b|\.o\.)"),
    # LLaMA-style MLP: gate_proj, up_proj, down_proj; GPT-NeoX: dense_h_to_4h, dense_4h_to_h
    "MLP": re.compile(r"(?:gate_proj|up_proj|down_proj|dense_h_to_4h|dense_4h_to_h|mlp\.)"),
    # layernorm variants
    "LayerNorm": re.compile(r"(?:layernorm|layer_norm|ln_|\.ln[0-9]*\b|^ln[0-9]*\b|input_layernorm|post_attention_layernorm)", re.IGNORECASE),
}

def bucket_of(name):
    for k, pat in PATTERNS.items():
        if pat.search(name):
            return k
    return "Other"

def layer_index_from_name(name):
    """
    Works for common HF layouts: 'model.layers.12.attn.q_proj.weight', 'transformer.h.12.attn.c_attn.weight', etc.
    Returns int or None if not found.
    """
    m = re.search(r"(?:layers|h)\.(\d+)\.", name)
    return int(m.group(1)) if m else None

def torch_load_any(path):
    """
    Loads a (possibly sharded) state_dict from a local path or HF repo id.
    Tries safetensors first if present.
    """
    # If it's a directory, try safetensors or pytorch_model.bin
    if os.path.isdir(path):
        # Merge sharded safetensors if any
        sd = {}
        sts = [f for f in os.listdir(path) if f.endswith(".safetensors")]
        if sts and safe_load:
            # may be multiple shards
            for f in sorted(sts):
                shard = safe_load(os.path.join(path, f))
                sd.update(shard)
            return sd
        # fallback to single bin
        bin_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            return torch.load(bin_path, map_location="cpu")
        # Some repos are named 'model.safetensors'
        single_st = os.path.join(path, "model.safetensors")
        if os.path.exists(single_st) and safe_load:
            return safe_load(single_st)
        raise FileNotFoundError(f"Could not find weights inside {path}")
    else:
        # Try to load from a single file
        if path.endswith(".safetensors") and safe_load:
            return safe_load(path)
        return torch.load(path, map_location="cpu")


def resolve_snapshot_dir(path):
    """Resolve Hugging Face cache layouts that keep weights inside snapshots/COMMITHASH."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return None

    snapshots_dir = os.path.join(path, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return path

    ref_main = os.path.join(path, "refs", "main")
    commit = None
    if os.path.isfile(ref_main):
        with open(ref_main, "r", encoding="utf-8") as f:
            commit = f.read().strip()
    if commit:
        resolved = os.path.join(snapshots_dir, commit)
        if os.path.isdir(resolved):
            return resolved

    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    if not snapshots:
        return None

    snapshots.sort(key=lambda d: os.path.getmtime(os.path.join(snapshots_dir, d)), reverse=True)
    return os.path.join(snapshots_dir, snapshots[0])


def align_state_dicts(sd_a, sd_b):
    # keep only common, identically-shaped params and exclude non-weights like buffers if desired
    keys = sorted(k for k in sd_a.keys() if k in sd_b and tuple(sd_a[k].shape) == tuple(sd_b[k].shape))
    return keys

def sparsity_of_delta(t, tau):
    if t.numel() == 0:
        return float("nan")
    return (t.abs() <= tau).float().mean().item() * 100.0

def is_linear_weight(name, tensor):
    return name.endswith("weight") and tensor.ndim == 2

def matrix_rank_percent(t, rtol=1e-5):
    # compute rank / min(m,n) * 100
    # For fp16/bf16, upcast for numerical stability
    tt = t.float()
    try:
        r = torch.linalg.matrix_rank(tt, rtol=rtol)
    except RuntimeError:
        # fallback via SVD (slower)
        u, s, v = torch.linalg.svd(tt, full_matrices=False)
        tol = rtol * s.max()
        r = (s > tol).sum()
    denom = min(tt.shape)
    return (r.item() / denom) * 100.0 if denom > 0 else float("nan")

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="Path to base checkpoint dir/file or repo id")
    ap.add_argument("--after",  required=True, help="Path to finetuned checkpoint dir/file or repo id")
    ap.add_argument("--tau", type=float, default=1e-6, help="magnitude threshold for |Δ| to count as zero")
    ap.add_argument("--rtol", type=float, default=1e-5, help="rank rtol for matrix_rank")
    ap.add_argument("--json_out", default="sparsity_report.json")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_out", default=None, help="Path to save the plot (e.g. sparsity_plot.png)")
    default_cache = "hf_cache" if os.path.isdir("hf_cache") else None
    ap.add_argument("--hf_cache_dir", default=default_cache, help="Optional Hugging Face cache directory to resolve repo ids")
    args = ap.parse_args()

    def resolve_checkpoint_spec(spec):
        candidates = []
        if os.path.exists(spec):
            candidates.append(spec)

        cache_root = args.hf_cache_dir
        if cache_root:
            cache_root = os.path.abspath(cache_root)
            if spec.startswith("models--"):
                candidates.append(os.path.join(cache_root, spec))
            if "/" in spec:
                repo_dir = f"models--{spec.replace('/', '--')}"
                candidates.append(os.path.join(cache_root, repo_dir))
            candidates.append(os.path.join(cache_root, spec))

        tried = set()
        for candidate in candidates:
            if candidate in tried:
                continue
            tried.add(candidate)
            resolved = resolve_snapshot_dir(candidate)
            if resolved and (os.path.isfile(resolved) or os.path.isdir(resolved)):
                return resolved

        raise FileNotFoundError(f"Could not resolve checkpoint spec '{spec}'" + (f" using cache root '{args.hf_cache_dir}'" if args.hf_cache_dir else ""))

    before_path = resolve_checkpoint_spec(args.before)
    after_path = resolve_checkpoint_spec(args.after)

    print("Loading checkpoints…")
    print(f"  before: {before_path}")
    print(f"  after : {after_path}")
    sd0 = torch_load_any(before_path)
    sd1 = torch_load_any(after_path)

    keys = align_state_dicts(sd0, sd1)
    if not keys:
        raise RuntimeError("No overlapping parameters with matching shapes were found.")

    per_layer_bucket = defaultdict(lambda: defaultdict(list))  # layer -> bucket -> [sparsity%]
    per_bucket_global = defaultdict(list)
    per_tensor = []

    rank_stats = []  # for linear weights

    for k in tqdm(keys):
        a = sd0[k].to(torch.float32)
        b = sd1[k].to(torch.float32)
        d = (b - a)

        sp = sparsity_of_delta(d, args.tau)
        layer = layer_index_from_name(k)
        bucket = bucket_of(k)

        per_tensor.append({"name": k, "layer": layer, "bucket": bucket, "sparsity_pct": sp})

        if layer is not None:
            per_layer_bucket[layer][bucket].append(sp)
            per_layer_bucket[layer]["Average"].append(sp)

        per_bucket_global[bucket].append(sp)
        per_bucket_global["Average"].append(sp)

        # matrix-rank check for linear weight updates
        if is_linear_weight(k, d):
            try:
                rank_pct = matrix_rank_percent(d, rtol=args.rtol)
                rank_stats.append({"name": k, "layer": layer, "bucket": bucket, "rank_pct": rank_pct})
            except Exception:
                pass

    # aggregate
    def agg_mean(dcts):
        return {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in dcts.items()}

    layerwise = {}
    for L, buckets in per_layer_bucket.items():
        layerwise[L] = agg_mean(buckets)

    global_means = agg_mean(per_bucket_global)

    report = {
        "threshold_tau": args.tau,
        "global_means_pct": global_means,
        "layerwise_means_pct": {int(k): v for k, v in sorted(layerwise.items(), key=lambda x: x[0])},
        "per_tensor": per_tensor,  # detailed table
        "rank_stats_pct": rank_stats,  # optional: linear weight Δ rank (% of full rank)
    }

    with open(args.json_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.json_out}")

    if args.plot or args.plot_out:
        import matplotlib
        if args.plot_out:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # collect series per bucket across layers
        all_layers = sorted(layerwise.keys())
        buckets = ["Average", "Q", "K", "V", "O", "MLP", "LayerNorm"]
        for title in ["Layerwise Sparsity"]:
            plt.figure(figsize=(9,4))
            for b in buckets:
                y = [layerwise[L].get(b, np.nan) for L in all_layers]
                plt.plot(all_layers, y, marker="o", label=b)
            plt.xlabel("Layer Index")
            plt.ylabel("Sparsity (%)")
            plt.ylim(0, 100)
            plt.title(title)
            plt.legend(ncol=3, fontsize=9)
            plt.tight_layout()
            if args.plot_out:
                plt.savefig(args.plot_out)
                print(f"Saved plot to {args.plot_out}")
            if args.plot:
                try:
                    plt.show()
                except Exception as e:
                    print(f"Could not show plot: {e}")

if __name__ == "__main__":
    main()
