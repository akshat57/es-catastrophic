import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.lines import Line2D

def load_report(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_sparsity_comparison():
    # Load data
    es_data = load_report('sparsity_report_es.json')
    grpo_data = load_report('sparsity_report_grpo.json')

    es_layers = es_data['layerwise_means_pct']
    grpo_layers = grpo_data['layerwise_means_pct']

    # Ensure layers are sorted ints
    layers_x = sorted([int(k) for k in es_layers.keys()])
    
    # Buckets to plot
    buckets = ["Average", "Q", "K", "V", "O", "MLP", "LayerNorm"]
    
    # Setup style
    sns.set_style("darkgrid")
    # Reduced height by 20% (3.5 * 0.8 = 2.8)
    fig, ax = plt.subplots(figsize=(3.25, 2.8))
    fig.subplots_adjust(top=0.72) 

    # Define colors for buckets
    # sns color palette
    palette = sns.color_palette()
    bucket_colors = {b: palette[i] for i, b in enumerate(buckets)}

    # Plot ES (Solid)
    for b in buckets:
        y_es = [es_layers[str(L)].get(b, np.nan) for L in layers_x]
        ax.plot(
            layers_x, 
            y_es, 
            linestyle="-", 
            linewidth=1.5, 
            color=bucket_colors[b],
        )

    # Plot GRPO (Dashed)
    for b in buckets:
        y_grpo = [grpo_layers[str(L)].get(b, np.nan) for L in layers_x]
        ax.plot(
            layers_x, 
            y_grpo, 
            linestyle="--", 
            linewidth=1.5, 
            color=bucket_colors[b],
        )

    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Sparsity (%)", fontsize=11)
    ax.set_ylim(0, 105) # a bit of headroom

    ax.tick_params(axis="both", labelsize=10)

    # Build legend handles
    bucket_elements = [
        Line2D([0], [0], color=bucket_colors[b], lw=1.5, label=b)
        for b in buckets
    ]
    method_elements = [
        Line2D([0], [0], color='black', lw=1.5, linestyle='-', label='ES'),
        Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='GRPO'),
    ]

    # Put Layer Type legend in figure coords (top center)
    leg_layer = fig.legend(
        handles=bucket_elements,
        title="Layer Type",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        fontsize=8,
        title_fontsize=8,
        handlelength=1.0,
        columnspacing=0.8,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )

    # Move Method legend inside the plot on the mid-right
    leg_method = ax.legend(
        handles=method_elements,
        title="Method",
        loc="center right",
        fontsize=8,
        title_fontsize=8,
        handlelength=1.5,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )

    out_path = "results/sparsity_comparison.png"
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=[leg_layer],  # leg_method is now inside ax
    )
    print(f"Saved {out_path}")

if __name__ == "__main__":
    plot_sparsity_comparison()