"""
probe_selector.py
-----------------
Selects the best (layer, probe_type) combination based on AUROC.
Produces a structured summary of all results for reporting.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProbeSelector:
    """
    Given the nested results dict from ProbeTrainer.train_all_layers(),
    finds the best layer and probe type by AUROC.

    Parameters
    ----------
    all_results : dict  layer_idx → probe_type → {"auroc", "f1", "accuracy", ...}
    """

    def __init__(self, all_results: Dict):
        self.all_results = all_results
        self._df: pd.DataFrame = self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        rows = []
        for layer_idx, probes in self.all_results.items():
            for probe_type, metrics in probes.items():
                rows.append({
                    "layer_idx"  : layer_idx,
                    "layer_label": f"L{layer_idx + 1:02d}",
                    "probe_type" : probe_type,
                    "auroc"      : metrics.get("auroc", np.nan),
                    "f1"         : metrics.get("f1", np.nan),
                    "accuracy"   : metrics.get("accuracy", np.nan),
                    "train_time_s": metrics.get("train_time_s", np.nan),
                })
        return pd.DataFrame(rows)

    @property
    def dataframe(self) -> pd.DataFrame:
        """All results as a tidy DataFrame."""
        return self._df

    def best(self) -> Tuple[int, str, float]:
        """
        Return (best_layer_idx, best_probe_type, best_auroc).
        """
        idx = self._df["auroc"].idxmax()
        row = self._df.loc[idx]
        logger.info(
            "Best probe: layer=%d (%s), type=%s, AUROC=%.4f",
            row["layer_idx"], row["layer_label"], row["probe_type"], row["auroc"],
        )
        return int(row["layer_idx"]), row["probe_type"], float(row["auroc"])

    def best_per_layer(self) -> pd.DataFrame:
        """Best probe type per layer (max AUROC)."""
        return (
            self._df.sort_values("auroc", ascending=False)
            .groupby("layer_idx")
            .first()
            .reset_index()
        )

    def peak_zone(self, top_n: int = 5) -> pd.DataFrame:
        """Top-n layers sorted by best AUROC."""
        return (
            self.best_per_layer()
            .sort_values("auroc", ascending=False)
            .head(top_n)
        )

    def summary(self) -> str:
        best_layer, best_probe, best_auroc = self.best()
        peak = self.peak_zone()
        lines = [
            "=" * 55,
            f"  Best AUROC : {best_auroc:.4f}",
            f"  Best layer : {best_layer + 1} (0-indexed: {best_layer})",
            f"  Best probe : {best_probe}",
            "",
            "  Top-5 layers by AUROC:",
            peak[["layer_label", "probe_type", "auroc"]].to_string(index=False),
            "=" * 55,
        ]
        return "\n".join(lines)
