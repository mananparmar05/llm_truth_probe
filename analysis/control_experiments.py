"""
control_experiments.py
-----------------------
Null-hypothesis control experiments (Phase 6 from PDF).

Three controls:
  1. Shuffled-label control — AUROC should ≈ 0.50
  2. Random-layer control   — probe trained on layer 0 AUROC ≈ 0.52
  3. Random baseline comparison — plotted on every ROC figure

These separate "I ran the code" from "I conducted a research experiment."
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)


class ControlExperiments:
    """
    Runs control experiments to guard against spurious findings.

    Parameters
    ----------
    hidden_states : np.ndarray  [N, num_layers, hidden_dim]
    labels        : np.ndarray  [N]
    n_repeats     : int  — number of random seeds for shuffled-label control
    """

    def __init__(
        self,
        hidden_states: np.ndarray,
        labels: np.ndarray,
        n_repeats: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.hidden_states = hidden_states
        self.labels = labels
        self.n_repeats = n_repeats
        self.test_size = test_size
        self.random_state = random_state
        self.num_layers = hidden_states.shape[1]

        n = len(labels)
        split = int(n * (1 - test_size))
        self.train_idx = np.arange(split)
        self.test_idx  = np.arange(split, n)

    def _make_probe(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=self.random_state)),
        ])

    # ── Control 1: Shuffled labels ────────────────────────────────────

    def shuffled_label_control(self, layer_idx: int = 14) -> Dict:
        """
        Train on hidden states at `layer_idx` but with randomly shuffled labels.
        Expected: AUROC ≈ 0.50 (chance).
        """
        X = self.hidden_states[:, layer_idx, :]
        X_train, X_test = X[self.train_idx], X[self.test_idx]
        y_test = self.labels[self.test_idx]

        aurocs = []
        for seed in range(self.n_repeats):
            y_train_shuffled = np.random.default_rng(seed).permuted(
                self.labels[self.train_idx]
            )
            probe = self._make_probe()
            probe.fit(X_train, y_train_shuffled)
            probs = probe.predict_proba(X_test)[:, 1]
            aurocs.append(float(roc_auc_score(y_test, probs)))

        result = {
            "control": "shuffled_labels",
            "layer_idx": layer_idx,
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "aurocs": aurocs,
        }
        logger.info(
            "Shuffled-label control @ layer %d: AUROC = %.4f ± %.4f",
            layer_idx, result["mean_auroc"], result["std_auroc"],
        )
        return result

    # ── Control 2: Random layer (layer 0) ────────────────────────────

    def random_layer_control(self, random_layer: int = 0) -> Dict:
        """
        Train probe on the first layer (near-random embeddings).
        Expected: AUROC ≈ 0.51–0.55.
        """
        X = self.hidden_states[:, random_layer, :]
        X_train, X_test = X[self.train_idx], X[self.test_idx]
        y_train = self.labels[self.train_idx]
        y_test  = self.labels[self.test_idx]

        probe = self._make_probe()
        probe.fit(X_train, y_train)
        probs = probe.predict_proba(X_test)[:, 1]
        auroc = float(roc_auc_score(y_test, probs))

        result = {
            "control": "random_layer",
            "layer_idx": random_layer,
            "auroc": auroc,
        }
        logger.info("Random-layer control @ layer %d: AUROC = %.4f", random_layer, auroc)
        return result

    # ── Control 3: All-layers control AUROC ──────────────────────────

    def layer_auroc_with_controls(self, real_aurocs: List[float]) -> Dict:
        """
        Compare real layer-wise AUROCs against shuffled and random baselines.

        Parameters
        ----------
        real_aurocs : list of float — actual AUROC per layer from ProbeTrainer
        """
        shuffled = self.shuffled_label_control(layer_idx=14)
        random_l = self.random_layer_control(random_layer=0)
        return {
            "real_aurocs"          : real_aurocs,
            "shuffled_mean_auroc"  : shuffled["mean_auroc"],
            "shuffled_std_auroc"   : shuffled["std_auroc"],
            "random_layer_auroc"   : random_l["auroc"],
        }

    # ── Plot ──────────────────────────────────────────────────────────

    @staticmethod
    def plot_control_comparison(
        real_aurocs: List[float],
        shuffled_mean: float,
        shuffled_std: float,
        random_auroc: float,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        n_layers = len(real_aurocs)
        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(range(1, n_layers + 1), real_aurocs,
                color="#2980b9", linewidth=2, label="Real probe (logistic)")
        ax.axhline(shuffled_mean, color="#e74c3c", linestyle="--", linewidth=1.5,
                   label=f"Shuffled labels: {shuffled_mean:.3f} ± {shuffled_std:.3f}")
        ax.axhspan(shuffled_mean - shuffled_std, shuffled_mean + shuffled_std,
                   alpha=0.15, color="#e74c3c")
        ax.axhline(random_auroc, color="#f39c12", linestyle=":", linewidth=1.5,
                   label=f"Random layer (L1): {random_auroc:.3f}")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)

        ax.set_xlabel("Layer (1-indexed)")
        ax.set_ylabel("AUROC")
        ax.set_title("AUROC vs Control Experiments", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0.45, 1.02)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Control comparison plot saved to %s", save_path)

        return fig
