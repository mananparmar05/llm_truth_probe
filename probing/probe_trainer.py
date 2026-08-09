"""
probe_trainer.py
----------------
Trains one probe classifier per transformer layer per probe type.
Probe types: LogisticRegression, MLP, SVM, Ensemble.

Output is a dict:
    results[layer_idx][probe_type] = {
        "probe"   : fitted sklearn estimator,
        "auroc"   : float,
        "f1"      : float,
        "accuracy": float,
    }

and optionally saves fitted probes via joblib.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Probe factory ─────────────────────────────────────────────────────

def _make_logistic_regression(C: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs",
                                   class_weight="balanced", random_state=42)),
    ])


def _make_mlp(hidden_layer_sizes: Tuple = (256, 64), max_iter: int = 500) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )),
    ])


def _make_svm(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=C, probability=True,
                    class_weight="balanced", random_state=42)),
    ])


PROBE_FACTORIES = {
    "logistic_regression": _make_logistic_regression,
    "mlp": _make_mlp,
    "svm": _make_svm,
}


# ── Main trainer ──────────────────────────────────────────────────────

class ProbeTrainer:
    """
    Trains lightweight probes on extracted hidden states.

    Parameters
    ----------
    hidden_states : np.ndarray  shape [N, num_layers, hidden_dim]
    labels        : np.ndarray  shape [N]
    probe_types   : list of probe names to train
    test_size     : fraction of data reserved for evaluation
    save_dir      : if provided, saves fitted probes here
    """

    def __init__(
        self,
        hidden_states: np.ndarray,
        labels: np.ndarray,
        probe_types: Optional[List[str]] = None,
        test_size: float = 0.2,
        save_dir: Optional[Path] = None,
        random_state: int = 42,
    ):
        self.hidden_states = hidden_states           # [N, L, H]
        self.labels = labels
        self.probe_types = probe_types or ["logistic_regression", "mlp", "svm"]
        self.test_size = test_size
        self.save_dir = Path(save_dir) if save_dir else None
        self.random_state = random_state

        self.num_layers = hidden_states.shape[1]
        self.hidden_dim = hidden_states.shape[2]

        # Train/test split (stratified)
        self.train_idx, self.test_idx = train_test_split(
            np.arange(len(labels)),
            test_size=test_size,
            stratify=labels,
            random_state=random_state,
        )
        logger.info(
            "ProbeTrainer — %d layers, %d probe types, train=%d, test=%d",
            self.num_layers, len(self.probe_types),
            len(self.train_idx), len(self.test_idx),
        )

    def _evaluate(self, probe, X_test, y_test) -> Dict:
        probs = probe.predict_proba(X_test)[:, 1]
        preds = probe.predict(X_test)
        return {
            "auroc"   : float(roc_auc_score(y_test, probs)),
            "f1"      : float(f1_score(y_test, preds, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, preds)),
        }

    def train_layer(self, layer_idx: int) -> Dict[str, Dict]:
        """Train all probe types for a single layer. Returns per-type metrics."""
        X = self.hidden_states[:, layer_idx, :]   # [N, H]
        X_train = X[self.train_idx]
        X_test  = X[self.test_idx]
        y_train = self.labels[self.train_idx]
        y_test  = self.labels[self.test_idx]

        layer_results: Dict[str, Dict] = {}
        for probe_type in self.probe_types:
            t0 = time.perf_counter()
            probe = PROBE_FACTORIES[probe_type]()
            probe.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0

            metrics = self._evaluate(probe, X_test, y_test)
            metrics["train_time_s"] = round(elapsed, 2)
            metrics["probe"] = probe

            layer_results[probe_type] = metrics
            logger.debug(
                "Layer %02d | %-22s | AUROC=%.4f | F1=%.4f | %.1fs",
                layer_idx, probe_type, metrics["auroc"], metrics["f1"], elapsed,
            )

        if self.save_dir:
            self._save_layer(layer_idx, layer_results)

        return layer_results

    def train_all_layers(self) -> Dict[int, Dict[str, Dict]]:
        """Train all probe types on all layers. Returns nested dict."""
        all_results: Dict[int, Dict[str, Dict]] = {}
        for layer_idx in tqdm(range(self.num_layers), desc="Training probes"):
            all_results[layer_idx] = self.train_layer(layer_idx)
        return all_results

    def build_ensemble(
        self,
        all_results: Dict[int, Dict[str, Dict]],
        top_k_layers: int = 5,
        ensemble_probe_type: str = "logistic_regression",
    ) -> Dict:
        """
        Build a layer-stacking ensemble from the top-k individual probes.

        Uses the trained logistic_regression probe from the best k layers
        and stacks their predicted probabilities via a meta-classifier.
        """
        # Rank layers by best single-probe AUROC
        layer_aurocs = [
            (layer_idx, max(res["auroc"] for res in probes.values()))
            for layer_idx, probes in all_results.items()
        ]
        top_layers = sorted(layer_aurocs, key=lambda x: x[1], reverse=True)[:top_k_layers]
        top_layer_indices = [l for l, _ in top_layers]

        logger.info("Ensemble top-%d layers: %s", top_k_layers, top_layer_indices)

        # Stack output probabilities from best probes
        X_meta_train = np.column_stack([
            all_results[l][ensemble_probe_type]["probe"].predict_proba(
                self.hidden_states[self.train_idx, l, :]
            )[:, 1]
            for l in top_layer_indices
        ])
        X_meta_test = np.column_stack([
            all_results[l][ensemble_probe_type]["probe"].predict_proba(
                self.hidden_states[self.test_idx, l, :]
            )[:, 1]
            for l in top_layer_indices
        ])

        meta_clf = LogisticRegression(max_iter=500, random_state=42)
        meta_clf.fit(X_meta_train, self.labels[self.train_idx])
        metrics = self._evaluate(meta_clf, X_meta_test, self.labels[self.test_idx])
        metrics["probe"] = meta_clf
        metrics["top_layer_indices"] = top_layer_indices
        logger.info("Ensemble AUROC=%.4f", metrics["auroc"])
        return metrics

    def _save_layer(self, layer_idx: int, layer_results: Dict) -> None:
        """Save fitted probes for one layer to disk."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for probe_type, res in layer_results.items():
            path = self.save_dir / f"probe_layer{layer_idx:02d}_{probe_type}.pkl"
            joblib.dump(res["probe"], path)

    @staticmethod
    def load_probe(path: Path):
        """Load a previously saved probe."""
        return joblib.load(path)
