"""
tsne_visualizer.py
------------------
t-SNE / PCA dimensionality reduction and scatter-plot visualisation
of hidden states, coloured by truthful / hallucinated label.

The primary output is `tsne_layer{N}.png` — the "money shot" for your
README, portfolio, and report.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

# Colour palette (matches the PDF code exactly)
_COLOUR_TRUTHFUL = "#2a78d6"
_COLOUR_HALLUC   = "#eb6834"


class TSNEVisualizer:
    """
    Reduces 4096-D hidden states to 2D via PCA → t-SNE and plots them.

    Parameters
    ----------
    hidden_states : np.ndarray  [N, hidden_dim]  (single layer already selected)
    labels        : np.ndarray  [N]  binary 0/1
    layer_idx     : int  — used for plot titles and filenames
    pca_components: int  — pre-reduce with PCA before t-SNE (speeds up greatly)
    tsne_perplexity, tsne_n_iter, random_state : standard t-SNE params
    """

    def __init__(
        self,
        hidden_states: np.ndarray,
        labels: np.ndarray,
        layer_idx: int = 14,
        pca_components: int = 50,
        tsne_perplexity: int = 40,
        tsne_n_iter: int = 1000,
        random_state: int = 42,
    ):
        self.hidden_states = hidden_states
        self.labels = labels
        self.layer_idx = layer_idx
        self.pca_components = pca_components
        self.tsne_perplexity = tsne_perplexity
        self.tsne_n_iter = tsne_n_iter
        self.random_state = random_state
        self._X_2d: Optional[np.ndarray] = None

    def _reduce(self) -> np.ndarray:
        """PCA → t-SNE pipeline."""
        logger.info("Running PCA(%d) …", self.pca_components)
        pca = PCA(n_components=min(self.pca_components, self.hidden_states.shape[1]),
                  random_state=self.random_state)
        X_pca = pca.fit_transform(self.hidden_states)

        logger.info("Running t-SNE(perplexity=%d, n_iter=%d) …",
                    self.tsne_perplexity, self.tsne_n_iter)
        tsne = TSNE(
            n_components=2,
            perplexity=self.tsne_perplexity,
            n_iter=self.tsne_n_iter,
            random_state=self.random_state,
        )
        return tsne.fit_transform(X_pca)

    def compute(self) -> np.ndarray:
        """Compute 2D embedding. Cached after first call."""
        if self._X_2d is None:
            self._X_2d = self._reduce()
        return self._X_2d

    def plot(
        self,
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Scatter plot coloured by label.

        Returns the matplotlib Figure so the caller can further customise.
        """
        X_2d = self.compute()
        layer_label = f"Layer {self.layer_idx + 1}"
        title = title or f"t-SNE projection of {layer_label} hidden states"

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            X_2d[self.labels == 0, 0], X_2d[self.labels == 0, 1],
            c=_COLOUR_TRUTHFUL, alpha=0.6, label="Truthful", s=20,
        )
        ax.scatter(
            X_2d[self.labels == 1, 0], X_2d[self.labels == 1, 1],
            c=_COLOUR_HALLUC, alpha=0.6, label="Hallucinated", s=20,
        )
        ax.legend(markerscale=2, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.grid(alpha=0.2)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("t-SNE plot saved to %s", save_path)

        return fig
