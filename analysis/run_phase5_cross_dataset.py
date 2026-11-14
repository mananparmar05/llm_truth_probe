"""
run_phase5_cross_dataset.py
----------------------------
Phase 5: Cross-Dataset Generalisation Study.
Tests whether truth/hallucination probes trained on HaluEval transfer
to TruthfulQA (and vice versa) across all 28 transformer layers.

Experiments:
  1. HaluEval   → HaluEval    (In-distribution baseline)
  2. TruthfulQA → TruthfulQA  (In-distribution baseline)
  3. HaluEval   → TruthfulQA  (Cross-dataset transfer)
  4. TruthfulQA → HaluEval    (Cross-dataset transfer)

Outputs:
  - results/cross_dataset_metrics.csv
  - results/plots/cross_dataset_generalization.png
"""

import sys
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
ROOT       = Path(__file__).resolve().parent.parent
H5_PATH    = ROOT / "data" / "hidden_states" / "hidden_states.h5"
RESULTS    = ROOT / "results"
PLOTS_DIR  = RESULTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE   = 0.20


def load_dataset(name: str):
    with h5py.File(H5_PATH, "r") as f:
        if name not in f:
            raise KeyError(f"Dataset '{name}' not found in {H5_PATH}. Available: {list(f.keys())}")
        hs     = f[name]["hidden_states"][:]
        labels = f[name]["labels"][:]
    return hs, labels


def split_data(hs: np.ndarray, labels: np.ndarray):
    n = len(labels)
    idx = np.arange(n)
    tr_idx, te_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
    )
    return {
        "X_train": hs[tr_idx],
        "y_train": labels[tr_idx],
        "X_test" : hs[te_idx],
        "y_test" : labels[te_idx],
    }


def fit_probe(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])
    clf.fit(X_train, y_train)
    return clf


def run_study(halu_data, truth_data, num_layers: int) -> pd.DataFrame:
    datasets = {
        "HaluEval"  : halu_data,
        "TruthfulQA": truth_data,
    }

    records = []
    logger.info("Running cross-dataset evaluation across all %d layers...", num_layers)

    for layer_idx in range(num_layers):
        for train_name, train_dict in datasets.items():
            X_tr = train_dict["X_train"][:, layer_idx, :]
            y_tr = train_dict["y_train"]

            probe = fit_probe(X_tr, y_tr)

            for test_name, test_dict in datasets.items():
                X_te = test_dict["X_test"][:, layer_idx, :]
                y_te = test_dict["y_test"]

                probs = probe.predict_proba(X_te)[:, 1]
                preds = (probs >= 0.5).astype(int)

                auroc = float(roc_auc_score(y_te, probs))
                acc   = float(accuracy_score(y_te, preds))
                f1    = float(f1_score(y_te, preds, zero_division=0))

                records.append({
                    "layer"      : layer_idx + 1,
                    "layer_idx"  : layer_idx,
                    "train_set"  : train_name,
                    "test_set"   : test_name,
                    "experiment" : f"{train_name} → {test_name}",
                    "split_type" : "In-Distribution" if train_name == test_name else "Cross-Dataset",
                    "auroc"      : auroc,
                    "accuracy"   : acc,
                    "f1"         : f1,
                })

        if (layer_idx + 1) % 4 == 0 or layer_idx == num_layers - 1:
            logger.info("Processed Layer %02d / %02d", layer_idx + 1, num_layers)

    df = pd.DataFrame(records)
    return df


def plot_generalization(df: pd.DataFrame, save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [2.2, 1]})

    palette = {
        "HaluEval → HaluEval": "#27ae60",
        "TruthfulQA → TruthfulQA": "#2ecc71",
        "HaluEval → TruthfulQA": "#e74c3c",
        "TruthfulQA → HaluEval": "#e67e22",
    }
    styles = {
        "In-Distribution": "-",
        "Cross-Dataset": "--",
    }

    # Left plot: AUROC per layer curve
    for exp_name, grp in df.groupby("experiment"):
        grp = grp.sort_values("layer")
        split_type = grp["split_type"].iloc[0]
        ax1.plot(
            grp["layer"],
            grp["auroc"],
            label=exp_name,
            color=palette.get(exp_name, "#333333"),
            linestyle=styles[split_type],
            linewidth=2.4 if split_type == "Cross-Dataset" else 1.8,
            marker="o" if split_type == "Cross-Dataset" else "s",
            markersize=4,
            alpha=0.9,
        )

    ax1.axhline(0.50, color="gray", linestyle=":", linewidth=1.2, label="Chance (0.50)")
    ax1.set_xlim(0.5, df["layer"].max() + 0.5)
    ax1.set_ylim(0.40, 1.02)
    ax1.set_xlabel("Transformer Layer (1-indexed)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Held-Out Test AUROC", fontsize=11, fontweight="bold")
    ax1.set_title("Cross-Dataset Generalisation Study — AUROC per Layer", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax1.grid(alpha=0.3)

    # Right plot: Mean AUROC comparison bar chart
    mean_df = df.groupby(["experiment", "split_type"])["auroc"].agg(["mean", "max"]).reset_index()
    y_pos = np.arange(len(mean_df))

    bars = ax2.barh(
        y_pos,
        mean_df["max"],
        color=[palette.get(e, "#333333") for e in mean_df["experiment"]],
        height=0.55,
        edgecolor="none",
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(mean_df["experiment"], fontsize=10)
    ax2.set_xlim(0.40, 1.02)
    ax2.set_xlabel("Peak Layer AUROC", fontsize=11, fontweight="bold")
    ax2.set_title("Peak Generalisation Comparison", fontsize=13, fontweight="bold")
    ax2.axvline(0.50, color="gray", linestyle=":", linewidth=1.2)
    ax2.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, mean_df["max"]):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", ha="left", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    logger.info("Saved generalization plot to %s", save_path)
    plt.close(fig)


def main():
    logger.info("Loading HaluEval and TruthfulQA from %s", H5_PATH)
    hs_halu, y_halu   = load_dataset("halueval")
    hs_truth, y_truth = load_dataset("truthfulqa")

    logger.info("HaluEval: %s | TruthfulQA: %s", hs_halu.shape, hs_truth.shape)
    num_layers = hs_halu.shape[1]

    halu_data  = split_data(hs_halu, y_halu)
    truth_data = split_data(hs_truth, y_truth)

    df = run_study(halu_data, truth_data, num_layers=num_layers)

    # Save metrics CSV
    csv_path = RESULTS / "cross_dataset_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved metrics to %s", csv_path)

    # Plot
    plot_path = PLOTS_DIR / "cross_dataset_generalization.png"
    plot_generalization(df, plot_path)

    # Print summary
    print("\n" + "=" * 65)
    print("      CROSS-DATASET GENERALISATION SUMMARY (PHASE 5)")
    print("=" * 65)
    summary = df.groupby("experiment").agg(
        peak_auroc=("auroc", "max"),
        mean_auroc=("auroc", "mean"),
    ).reset_index()
    for _, row in summary.iterrows():
        print(f"  {row['experiment']:<26} | Peak AUROC: {row['peak_auroc']:.4f} | Mean AUROC: {row['mean_auroc']:.4f}")
    print("=" * 65)
    print(f"\n📊 Metrics saved: {csv_path}")
    print(f"📈 Plot saved   : {plot_path}\n")


if __name__ == "__main__":
    main()

