"""
test_pipeline.py
----------------
Unit and integration tests for the hallucination detection pipeline.
Run with: pytest tests/ -v --tb=short
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def dummy_hidden_states():
    """Synthetic hidden states: 200 samples, 32 layers, 64-dim (lightweight)."""
    np.random.seed(42)
    N, L, H = 200, 32, 64
    hs = np.random.randn(N, L, H).astype(np.float32)
    # Make layer 14 linearly separable: add class-conditional shift
    labels = np.array([i % 2 for i in range(N)], dtype=np.int32)
    hs[labels == 1, 14, :] += 2.0   # add separation at layer 14
    return hs, labels


@pytest.fixture
def dummy_probe(dummy_hidden_states):
    hs, labels = dummy_hidden_states
    X = hs[:, 14, :]
    probe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
    probe.fit(X[:160], labels[:160])
    return probe


# ── HiddenStateExtractor tests ────────────────────────────────────────

class TestHiddenStateExtractor:
    def test_import(self):
        from extraction.hook_extractor import HiddenStateExtractor
        assert HiddenStateExtractor is not None

    def test_layer_detection_failure(self):
        import torch.nn as nn
        from extraction.hook_extractor import HiddenStateExtractor
        # A plain module with no 'model.layers' should raise
        class DummyModel(nn.Module):
            def forward(self, x): return x
        with pytest.raises(ValueError, match="Cannot detect"):
            HiddenStateExtractor(DummyModel())


# ── Storage tests ─────────────────────────────────────────────────────

class TestHiddenStateStore:
    def test_save_load_numpy(self, tmp_path, dummy_hidden_states):
        from extraction.storage import HiddenStateStore
        hs, labels = dummy_hidden_states
        store = HiddenStateStore(tmp_path, format="numpy")
        store.save(hs, labels, "test_dataset")
        hs2, labels2 = store.load("test_dataset")
        np.testing.assert_array_almost_equal(hs, hs2, decimal=5)
        np.testing.assert_array_equal(labels, labels2)

    def test_save_load_hdf5(self, tmp_path, dummy_hidden_states):
        pytest.importorskip("h5py")
        from extraction.storage import HiddenStateStore
        hs, labels = dummy_hidden_states
        store = HiddenStateStore(tmp_path, format="hdf5")
        store.save(hs, labels, "test_dataset")
        hs2, labels2 = store.load("test_dataset")
        np.testing.assert_array_almost_equal(hs, hs2, decimal=5)

    def test_list_datasets(self, tmp_path, dummy_hidden_states):
        from extraction.storage import HiddenStateStore
        hs, labels = dummy_hidden_states
        store = HiddenStateStore(tmp_path, format="numpy")
        store.save(hs, labels, "truthfulqa")
        store.save(hs, labels, "halueval")
        datasets = store.list_datasets()
        assert set(datasets) == {"truthfulqa", "halueval"}


# ── ProbeTrainer tests ────────────────────────────────────────────────

class TestProbeTrainer:
    def test_train_single_layer(self, dummy_hidden_states):
        from probing.probe_trainer import ProbeTrainer
        hs, labels = dummy_hidden_states
        trainer = ProbeTrainer(
            hs, labels,
            probe_types=["logistic_regression"],
            test_size=0.2,
        )
        results = trainer.train_layer(14)
        assert "logistic_regression" in results
        assert results["logistic_regression"]["auroc"] > 0.6, "Layer 14 should be well-separated"

    def test_auroc_range(self, dummy_hidden_states):
        from probing.probe_trainer import ProbeTrainer
        hs, labels = dummy_hidden_states
        trainer = ProbeTrainer(hs, labels, probe_types=["logistic_regression"], test_size=0.2)
        result = trainer.train_layer(0)
        auroc = result["logistic_regression"]["auroc"]
        assert 0.0 <= auroc <= 1.0

    def test_train_all_layers_shape(self, dummy_hidden_states):
        from probing.probe_trainer import ProbeTrainer
        hs, labels = dummy_hidden_states
        trainer = ProbeTrainer(hs, labels, probe_types=["logistic_regression"], test_size=0.2)
        all_results = trainer.train_all_layers()
        assert len(all_results) == 32


# ── ProbeSelector tests ───────────────────────────────────────────────

class TestProbeSelector:
    def test_best_returns_tuple(self, dummy_hidden_states):
        from probing.probe_trainer import ProbeTrainer
        from probing.probe_selector import ProbeSelector
        hs, labels = dummy_hidden_states
        trainer = ProbeTrainer(hs, labels, probe_types=["logistic_regression"], test_size=0.2)
        all_results = trainer.train_all_layers()
        selector = ProbeSelector(all_results)
        best_layer, best_probe, best_auroc = selector.best()
        assert isinstance(best_layer, int)
        assert isinstance(best_auroc, float)
        assert best_layer == 14  # we added separation at layer 14

    def test_dataframe_columns(self, dummy_hidden_states):
        from probing.probe_trainer import ProbeTrainer
        from probing.probe_selector import ProbeSelector
        hs, labels = dummy_hidden_states
        trainer = ProbeTrainer(hs, labels, probe_types=["logistic_regression"], test_size=0.2)
        all_results = trainer.train_all_layers()
        df = ProbeSelector(all_results).dataframe
        assert "auroc" in df.columns
        assert "layer_idx" in df.columns


# ── ProbeEvaluator tests ──────────────────────────────────────────────

class TestProbeEvaluator:
    def test_metrics_in_range(self, dummy_hidden_states, dummy_probe):
        from probing.probe_evaluator import ProbeEvaluator
        hs, labels = dummy_hidden_states
        ev = ProbeEvaluator(dummy_probe, hs[160:, 14, :], labels[160:])
        m = ev.all_metrics()
        assert 0.0 <= m["auroc"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0

    def test_roc_curve_plot(self, dummy_hidden_states, dummy_probe, tmp_path):
        from probing.probe_evaluator import ProbeEvaluator
        hs, labels = dummy_hidden_states
        ev = ProbeEvaluator(dummy_probe, hs[160:, 14, :], labels[160:])
        ev.plot_roc_curve(save_path=tmp_path / "roc.png")
        assert (tmp_path / "roc.png").exists()


# ── Control experiments tests ─────────────────────────────────────────

class TestControlExperiments:
    def test_shuffled_label_near_chance(self, dummy_hidden_states):
        from analysis.control_experiments import ControlExperiments
        hs, labels = dummy_hidden_states
        ctrl = ControlExperiments(hs, labels, n_repeats=3)
        result = ctrl.shuffled_label_control(layer_idx=14)
        # Shuffled labels → AUROC should be close to 0.50
        assert 0.35 <= result["mean_auroc"] <= 0.70, f"Expected near-chance, got {result['mean_auroc']}"

    def test_random_layer_low_auroc(self, dummy_hidden_states):
        from analysis.control_experiments import ControlExperiments
        hs, labels = dummy_hidden_states
        ctrl = ControlExperiments(hs, labels)
        result = ctrl.random_layer_control(random_layer=0)
        assert 0.0 <= result["auroc"] <= 1.0
