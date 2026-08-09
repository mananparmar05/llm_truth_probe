"""
cross_dataset.py
----------------
Cross-dataset generalisation study:
  Train on TruthfulQA → evaluate on HaluEval (and vice versa).

This is the original contribution described in the PDF (Section 5/9).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CrossDatasetStudy:
    """
    Systematic 4-experiment generalisation study.

    Experiments
    -----------
    1. TruthfulQA → TruthfulQA  (same-distribution baseline)
    2. HaluEval   → HaluEval    (same-distribution baseline)
    3. TruthfulQA → HaluEval    (cross-dataset generalisation)
    4. HaluEval   → TruthfulQA  (cross-dataset generalisation)

    Each experiment is run for every transformer layer so we can
    plot cross-dataset AUROC as a function of layer depth.
    """

    def __init__(
        self,
        datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        layer_indices: list | None = None,
    ):
        """
        Parameters
        ----------
        datasets     : dict  name → (hidden_states [N,L,H], labels [N])
        layer_indices: which layers to evaluate; None → all
        """
        self.datasets = datasets
        first_hs = next(iter(datasets.values()))[0]
        self.num_layers = first_hs.shape[1]
        self.layer_indices = layer_indices or list(range(self.num_layers))

    def _fit_probe(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        probe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42,
                                       class_weight="balanced")),
        ])
        probe.fit(X_train, y_train)
        return probe

    def run_layer(self, layer_idx: int) -> pd.DataFrame:
        """Run all 4 experiments for a single layer."""
        rows = []
        for train_name, (hs_train, y_train) in self.datasets.items():
            probe = self._fit_probe(hs_train[:, layer_idx, :], y_train)
            for test_name, (hs_test, y_test) in self.datasets.items():
                probs = probe.predict_proba(hs_test[:, layer_idx, :])[:, 1]
                auroc = float(roc_auc_score(y_test, probs))
                rows.append({
                    "layer_idx"  : layer_idx,
                    "train_set"  : train_name,
                    "test_set"   : test_name,
                    "auroc"      : auroc,
                    "split_type" : "same-dist" if train_name == test_name else "cross-dataset",
                })
        return pd.DataFrame(rows)

    def run_all(self) -> pd.DataFrame:
        """Run all experiments for all layers. Returns tidy DataFrame."""
        dfs = []
        for layer_idx in self.layer_indices:
            dfs.append(self.run_layer(layer_idx))
            logger.debug("Cross-dataset layer %d done.", layer_idx)
        df = pd.concat(dfs, ignore_index=True)
        logger.info("Cross-dataset study complete — %d rows", len(df))
        return df

    @staticmethod
    def plot(df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
        """
        Line plot: layer index on x-axis, AUROC on y-axis,
        one line per (train_set, test_set) combination.
        """
        fig, ax = plt.subplots(figsize=(13, 6))
        combos = df.groupby(["train_set", "test_set"])

        palette = {
            "same-dist"   : "#2ecc71",
            "cross-dataset": "#e74c3c",
        }
        linestyles = {"same-dist": "-", "cross-dataset": "--"}

        for (train, test), group in combos:
            split_type = "same-dist" if train == test else "cross-dataset"
            g = group.sort_values("layer_idx")
            ax.plot(
                g["layer_idx"] + 1,
                g["auroc"],
                label=f"{train} → {test}",
                color=palette[split_type],
                linestyle=linestyles[split_type],
                linewidth=2,
                alpha=0.85,
            )

        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Layer (1-indexed)", fontsize=11)
        ax.set_ylabel("AUROC", fontsize=11)
        ax.set_title("Cross-Dataset Generalisation Study — AUROC per Layer", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, ncol=2)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.45, 1.02)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Cross-dataset plot saved to %s", save_path)

        return fig
