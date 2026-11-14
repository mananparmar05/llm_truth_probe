"""
run_phase4_viz.py
-----------------
Phase 4: Visualizations
Loads saved probe .pkl files + hidden_states.h5, re-evaluates every probe on
the held-out test split, and produces:
  1. AUROC bar chart per probe type (SVM / MLP / LR)
  2. Probe-type comparison heatmap (layers × probe types)
  3. Top-5 layer summary bar chart
Saves all plots to results/plots/
"""

import sys
import logging
from pathlib import Path

import joblib
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
H5_PATH    = ROOT / "data" / "hidden_states" / "hidden_states.h5"
MODELS_DIR = ROOT / "results" / "models"
PLOTS_DIR  = ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PROBE_TYPES  = ["svm", "mlp", "logistic_regression"]
PROBE_LABELS = {"svm": "RBF SVM", "mlp": "MLP (256,64)", "logistic_regression": "Logistic Regression"}
PROBE_COLORS = {"svm": "#e74c3c", "mlp": "#3498db", "logistic_regression": "#2ecc71"}
RANDOM_SEED  = 42
TEST_SIZE    = 0.20   # same split used in ProbeTrainer


def load_hidden_states():
    logger.info("Loading hidden states from %s", H5_PATH)
    with h5py.File(H5_PATH, "r") as f:
        keys = list(f.keys())
        logger.info("Available datasets: %s", keys)
        # Prefer halueval (the 20k run), fall back to first available
        ds = "halueval" if "halueval" in keys else keys[0]
        hs     = f[ds]["hidden_states"][:]   # (N, L, H)
        labels = f[ds]["labels"][:]
    logger.info("Loaded hs shape: %s, labels shape: %s", hs.shape, labels.shape)
    return hs, labels


def get_test_split(hs, labels):
    """Reproduce the exact same 80/20 stratified split used during training."""
    n = len(labels)
    idx = np.arange(n)
    _, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
    )
    return test_idx


def evaluate_all_probes(hs, labels, test_idx):
    """Load every saved probe and compute AUROC on the held-out test set."""
    n_layers  = hs.shape[1]
    y_test    = labels[test_idx]
    results   = {pt: [] for pt in PROBE_TYPES}

    for layer_idx in range(n_layers):
        X_test = hs[test_idx, layer_idx, :]
        for pt in PROBE_TYPES:
            pkl = MODELS_DIR / f"probe_layer{layer_idx:02d}_{pt}.pkl"
            if not pkl.exists():
                logger.warning("Missing: %s — filling with NaN", pkl.name)
                results[pt].append(float("nan"))
                continue
            probe = joblib.load(pkl)
            proba = probe.predict_proba(X_test)[:, 1]
            auroc = float(roc_auc_score(y_test, proba))
            results[pt].append(auroc)
        logger.info("Layer %02d evaluated.", layer_idx)

    return results   # dict: probe_type → list[float] length=n_layers


# ── Plot 1: Per-probe AUROC bar chart ────────────────────────────────────────

def plot_auroc_bars(results, n_layers, probe_type, save_name):
    aurocs = results[probe_type]
    peak   = int(np.nanargmax(aurocs))

    colours = []
    for i in range(n_layers):
        if i == peak:
            colours.append("#f39c12")          # gold = best
        elif i in [4, 5, 6]:                   # known top-SVM zone
            colours.append("#e74c3c")
        elif i in [14, 15]:                    # known top-MLP zone
            colours.append("#3498db")
        else:
            colours.append("#95a5a6")

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(n_layers), aurocs, color=colours, edgecolor="none", width=0.75)
    ax.axhline(0.50, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Random (0.50)")
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(0.45, 1.01)
    ax.set_xlabel("Transformer Layer", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title(
        f"Layer-wise AUROC — {PROBE_LABELS[probe_type]}  |  HaluEval (N=20,000)  |  Qwen2.5-7B",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i+1:02d}" for i in range(n_layers)], rotation=45, ha="right", fontsize=8)

    # Annotate peak
    ax.annotate(
        f"Peak L{peak+1:02d}\nAUROC={aurocs[peak]:.4f}",
        xy=(peak, aurocs[peak]),
        xytext=(peak + 2 if peak < n_layers - 4 else peak - 5, aurocs[peak] - 0.04),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9, fontweight="bold",
    )

    patches = [
        mpatches.Patch(color="#f39c12", label=f"Peak layer (L{peak+1:02d})"),
        mpatches.Patch(color="#e74c3c", label="SVM peak zone (L5–L7)"),
        mpatches.Patch(color="#3498db", label="MLP peak zone (L15–L16)"),
        mpatches.Patch(color="#95a5a6", label="Other layers"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = PLOTS_DIR / save_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", out)
    plt.close(fig)


# ── Plot 2: Heatmap (probe types × layers) ───────────────────────────────────

def plot_heatmap(results, n_layers):
    import pandas as pd
    df = pd.DataFrame(
        {PROBE_LABELS[pt]: results[pt] for pt in PROBE_TYPES},
        index=[f"L{i+1:02d}" for i in range(n_layers)],
    ).T   # rows=probes, cols=layers

    fig, ax = plt.subplots(figsize=(22, 3))
    sns.heatmap(
        df, ax=ax,
        cmap="RdYlGn", vmin=0.50, vmax=1.0,
        annot=True, fmt=".3f", annot_kws={"size": 6.5},
        linewidths=0.3, linecolor="white",
    )
    ax.set_title(
        "AUROC Heatmap — All Probes × All Layers  |  HaluEval (N=20,000)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Type")
    fig.tight_layout()

    out = PLOTS_DIR / "heatmap_probe_vs_layer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", out)
    plt.close(fig)


# ── Plot 3: Top-5 comparison bar chart ───────────────────────────────────────

def plot_top5_summary(results, n_layers):
    rows = []
    for pt in PROBE_TYPES:
        for i, auc in enumerate(results[pt]):
            rows.append((f"L{i+1:02d} / {PROBE_LABELS[pt]}", auc, pt))

    rows.sort(key=lambda x: x[1], reverse=True)
    top5 = rows[:5]

    labels_  = [r[0] for r in top5]
    aurocs_  = [r[1] for r in top5]
    colors_  = [PROBE_COLORS[r[2]] for r in top5]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels_[::-1], aurocs_[::-1], color=colors_[::-1], edgecolor="none", height=0.5)
    ax.set_xlim(0.978, 0.985)
    ax.set_xlabel("AUROC", fontsize=11)
    ax.set_title("Top-5 Probe Configurations  |  HaluEval (N=20,000)", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, aurocs_[::-1]):
        ax.text(val + 0.00005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9, fontweight="bold")
    patches = [mpatches.Patch(color=PROBE_COLORS[pt], label=PROBE_LABELS[pt]) for pt in PROBE_TYPES]
    ax.legend(handles=patches, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = PLOTS_DIR / "top5_probes_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", out)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    hs, labels = load_hidden_states()
    n_layers   = hs.shape[1]
    test_idx   = get_test_split(hs, labels)

    logger.info("Evaluating all 84 probes on %d test samples...", len(test_idx))
    results = evaluate_all_probes(hs, labels, test_idx)

    logger.info("Generating plots...")
    plot_auroc_bars(results, n_layers, "svm",                   "auroc_bars_svm.png")
    plot_auroc_bars(results, n_layers, "mlp",                   "auroc_bars_mlp.png")
    plot_auroc_bars(results, n_layers, "logistic_regression",   "auroc_bars_lr.png")
    plot_heatmap(results, n_layers)
    plot_top5_summary(results, n_layers)

    logger.info("✅ Phase 4 complete! All plots saved to: %s", PLOTS_DIR)
    print(f"\n📊 Plots saved to: {PLOTS_DIR}")
    print("   • auroc_bars_svm.png")
    print("   • auroc_bars_mlp.png")
    print("   • auroc_bars_lr.png")
    print("   • heatmap_probe_vs_layer.png")
    print("   • top5_probes_summary.png")


if __name__ == "__main__":
    main()

