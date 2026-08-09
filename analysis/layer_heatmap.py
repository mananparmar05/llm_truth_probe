"""
layer_heatmap.py
----------------
Generates layer-wise AUROC bar charts and cross-model heatmaps.

Key plots produced:
  1. Bar chart: AUROC per layer for one model/dataset/probe.
  2. Heatmap: model × layer AUROC comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class LayerHeatmap:
    """
    Visualises layer-wise AUROC across models, datasets, and probe types.
    """

    @staticmethod
    def plot_auroc_bar(
        layer_aurocs: List[float],
        probe_type: str = "logistic_regression",
        dataset_name: str = "TruthfulQA",
        model_name: str = "Llama-3.1-8B",
        save_path: Optional[Path] = None,
        highlight_layers: Optional[List[int]] = None,
    ) -> plt.Figure:
        """
        Bar chart with AUROC on y-axis and layer index on x-axis.
        Highlights the peak zone (layers 14–16 by default) in green.

        Parameters
        ----------
        layer_aurocs     : list of AUROC values, one per layer (0-indexed)
        highlight_layers : 0-indexed layer indices to colour green
        """
        n_layers = len(layer_aurocs)
        highlight_layers = highlight_layers or list(range(13, 16))  # layers 14-16

        colours = [
            "#27ae60" if i in highlight_layers else "#3498db"
            for i in range(n_layers)
        ]

        fig, ax = plt.subplots(figsize=(14, 5))
        bars = ax.bar(range(n_layers), layer_aurocs, color=colours, edgecolor="none", width=0.8)

        ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Random (0.50)")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(0.45, 1.0)
        ax.set_xlabel("Transformer Layer Index (1-indexed)", fontsize=11)
        ax.set_ylabel("AUROC", fontsize=11)
        ax.set_title(
            f"Layer-wise AUROC — {model_name} | {dataset_name} | {probe_type}",
            fontsize=12, fontweight="bold",
        )
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"L{i+1}" for i in range(n_layers)], rotation=45, ha="right", fontsize=8)

        # Annotate peak
        peak_layer = int(np.argmax(layer_aurocs))
        ax.annotate(
            f"Peak L{peak_layer + 1}\nAUROC={layer_aurocs[peak_layer]:.3f}",
            xy=(peak_layer, layer_aurocs[peak_layer]),
            xytext=(peak_layer + 2, layer_aurocs[peak_layer] - 0.05),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )

        # Legend for highlight
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#27ae60", label="Peak zone (L14–L16)"),
            Patch(facecolor="#3498db", label="Other layers"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("AUROC bar chart saved to %s", save_path)

        return fig

    @staticmethod
    def plot_model_comparison_heatmap(
        model_aurocs: Dict[str, List[float]],
        dataset_name: str = "TruthfulQA",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Heatmap: rows = models, columns = layers, values = AUROC.
        Compares Llama-3.1-8B vs. Mistral-7B side by side.

        Parameters
        ----------
        model_aurocs : dict model_name → list[float] (one per layer)
        """
        df = pd.DataFrame(model_aurocs).T  # rows=models, cols=layers
        df.columns = [f"L{i+1}" for i in range(df.shape[1])]

        fig, ax = plt.subplots(figsize=(16, 3))
        sns.heatmap(
            df,
            ax=ax,
            cmap="RdYlGn",
            vmin=0.5,
            vmax=1.0,
            annot=(df.shape[1] <= 32),
            fmt=".2f",
            annot_kws={"size": 7},
            linewidths=0.3,
        )
        ax.set_title(
            f"AUROC Heatmap — {dataset_name} — Model Comparison",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Model")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Model comparison heatmap saved to %s", save_path)

        return fig
