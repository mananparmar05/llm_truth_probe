"""
run_training.py
---------------
CLI runner script for Phase 3: Probe Training.
Loads hidden state vectors from HDF5, trains Logistic Regression, MLP, and SVM probes
across all layers, constructs layer-stacking ensemble, and saves fitted models.
"""

import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from extraction.storage import HiddenStateStore
from probing.probe_trainer import ProbeTrainer
from probing.probe_selector import ProbeSelector


def main():
    store_dir = project_root / "data" / "hidden_states"
    store = HiddenStateStore(store_dir, format="hdf5")
    
    datasets = store.list_datasets()
    if not datasets:
        logger.error(f"No datasets found in {store.path}. Please verify the file exists!")
        sys.exit(1)

    logger.info(f"Available datasets in HDF5 store: {datasets}")
    # Prefer truthfulqa for quick training, or halueval with fast probes
    dataset_name = "truthfulqa" if "truthfulqa" in datasets else datasets[0]
    
    logger.info(f"Loading hidden states for dataset '{dataset_name}'...")
    hs, labels = store.load(dataset_name)
    logger.info(f"Loaded tensor shape: {hs.shape} (N_samples={hs.shape[0]}, Num_layers={hs.shape[1]}, Hidden_dim={hs.shape[2]})")

    models_dir = project_root / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # For large datasets (N > 5000), RBF SVM has quadratic complexity O(N^2).
    # Use Logistic Regression & MLP which scale linearly O(N) and finish in ~1-2 minutes!
    probe_types = ["logistic_regression", "mlp"] if hs.shape[0] > 5000 else ["logistic_regression", "mlp", "svm"]

    logger.info(f"Starting probe training across all layers using probes: {probe_types}...")
    trainer = ProbeTrainer(
        hidden_states=hs,
        labels=labels,
        probe_types=probe_types,
        save_dir=models_dir,
    )

    all_results = trainer.train_all_layers()

    # Build Layer Stacking Ensemble
    logger.info("Building top-layer stacking ensemble probe...")
    ensemble_results = trainer.build_ensemble(all_results, top_k_layers=5)

    # Summarize & select best probe
    selector = ProbeSelector(all_results)
    print("\n" + selector.summary())
    
    print(f"\n[Ensemble Model Results]")
    print(f"  Top-5 Layer Stacking Ensemble AUROC : {ensemble_results['auroc']:.4f}")
    print(f"  Accuracy                            : {ensemble_results['accuracy']:.4f}")
    print(f"  F1-Score                            : {ensemble_results['f1']:.4f}")

    print(f"\n✅ Training complete! Fitted probes saved to: {models_dir}")


if __name__ == "__main__":
    main()
