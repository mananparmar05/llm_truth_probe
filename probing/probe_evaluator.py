"""
probe_evaluator.py
------------------
Full evaluation suite: AUROC, F1, accuracy, confusion matrix, ROC curve.
Also provides cross-dataset evaluation (train on A, test on B).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ProbeEvaluator:
    """
    Evaluates a trained probe on a hidden-state test set.

    Parameters
    ----------
    probe        : fitted sklearn estimator (must support predict_proba)
    X_test       : np.ndarray  [N_test, hidden_dim]
    y_test       : np.ndarray  [N_test]
    """

    def __init__(self, probe, X_test: np.ndarray, y_test: np.ndarray):
        self.probe = probe
        self.X_test = X_test
        self.y_test = y_test
        self._probs: Optional[np.ndarray] = None
        self._preds: Optional[np.ndarray] = None

    def _ensure_predictions(self):
        if self._probs is None:
            self._probs = self.probe.predict_proba(self.X_test)[:, 1]
            self._preds = self.probe.predict(self.X_test)

    # ── Scalar metrics ────────────────────────────────────────────────

    def auroc(self) -> float:
        self._ensure_predictions()
        return float(roc_auc_score(self.y_test, self._probs))

    def f1(self) -> float:
        self._ensure_predictions()
        return float(f1_score(self.y_test, self._preds, zero_division=0))

    def accuracy(self) -> float:
        self._ensure_predictions()
        return float(accuracy_score(self.y_test, self._preds))

    def all_metrics(self) -> Dict[str, float]:
        return {
            "auroc"   : self.auroc(),
            "f1"      : self.f1(),
            "accuracy": self.accuracy(),
        }

    # ── Plots ─────────────────────────────────────────────────────────

    def plot_roc_curve(
        self,
        label: str = "Probe",
        save_path: Optional[Path] = None,
        ax=None,
    ) -> None:
        self._ensure_predictions()
        fpr, tpr, _ = roc_curve(self.y_test, self._probs)
        roc_auc = auc(fpr, tpr)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Hallucination Probe")
        ax.legend()
        ax.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("ROC curve saved to %s", save_path)

    def plot_confusion_matrix(
        self,
        save_path: Optional[Path] = None,
        ax=None,
    ) -> None:
        self._ensure_predictions()
        cm = confusion_matrix(self.y_test, self._preds)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Truthful", "Hallucinated"],
        )
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Confusion matrix saved to %s", save_path)


# ── Cross-dataset evaluator ───────────────────────────────────────────

class CrossDatasetEvaluator:
    """
    Runs the four generalization experiments:
      TruthfulQA → TruthfulQA  (baseline)
      HaluEval   → HaluEval    (baseline)
      TruthfulQA → HaluEval    (cross-dataset)
      HaluEval   → TruthfulQA  (cross-dataset)

    Parameters
    ----------
    trainer_cls  : ProbeTrainer class (not instance)
    datasets     : dict name → (hidden_states [N,L,H], labels [N])
    layer_idx    : layer to use (e.g. 14 for layer 15, 0-indexed)
    probe_type   : which probe to use ("logistic_regression" recommended)
    """

    def __init__(self, trainer_cls, datasets: Dict, layer_idx: int = 14, probe_type: str = "logistic_regression"):
        self.trainer_cls = trainer_cls
        self.datasets = datasets
        self.layer_idx = layer_idx
        self.probe_type = probe_type

    def run(self) -> Dict:
        results = {}
        dataset_names = list(self.datasets.keys())

        for train_name in dataset_names:
            hs_train, y_train = self.datasets[train_name]
            X_train = hs_train[:, self.layer_idx, :]

            trainer = self.trainer_cls(
                hidden_states=hs_train,
                labels=y_train,
                probe_types=[self.probe_type],
                test_size=0.0,        # use full dataset for training in cross-eval
            )
            # Manually fit probe
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            probe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ])
            probe.fit(X_train, y_train)

            for test_name in dataset_names:
                hs_test, y_test = self.datasets[test_name]
                X_test = hs_test[:, self.layer_idx, :]
                evaluator = ProbeEvaluator(probe, X_test, y_test)
                key = f"{train_name}_→_{test_name}"
                results[key] = evaluator.all_metrics()
                logger.info(
                    "Cross-eval %s: AUROC=%.4f", key, results[key]["auroc"]
                )

        return results
