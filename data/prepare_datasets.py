"""
prepare_datasets.py
-------------------
Downloads TruthfulQA and HaluEval benchmarks from their original
GitHub repositories, formats them into a uniform (text, label) schema,
performs stratified 80/10/10 splits, and saves the results as CSV files.

This script requires NO GPU, NO HuggingFace access — it downloads
raw data from GitHub using only stdlib + pandas + sklearn.

Usage:
    python data/prepare_datasets.py

Output:
    data/truthfulqa/{train,val,test}.csv
    data/halueval/{train,val,test}.csv
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Resolve project root (one level up from data/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ── Source URLs (GitHub raw files) ────────────────────────────────────
TRUTHFULQA_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
HALUEVAL_QA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"


# ── SSL context ───────────────────────────────────────────────────────

def _get_ssl_context() -> ssl.SSLContext:
    """
    Build an SSL context that works on corporate macOS machines.

    Strategy:
      1. Try macOS system keychain certs (includes corporate CAs)
      2. Fall back to certifi bundle
      3. Fall back to default context
    """
    system_certs = Path("/tmp/system_certs.pem")

    # Try to export macOS system certs if not already done
    if not system_certs.exists():
        try:
            import subprocess
            with open(system_certs, "w") as f:
                for keychain in [
                    "/System/Library/Keychains/SystemRootCertificates.keychain",
                    "/Library/Keychains/System.keychain",
                    os.path.expanduser("~/Library/Keychains/login.keychain-db"),
                ]:
                    try:
                        result = subprocess.run(
                            ["security", "find-certificate", "-a", "-p", keychain],
                            capture_output=True, text=True, timeout=10,
                        )
                        f.write(result.stdout)
                    except Exception:
                        pass
            logger.debug("Exported macOS system certs to %s", system_certs)
        except Exception:
            pass

    # Try system certs first
    if system_certs.exists() and system_certs.stat().st_size > 0:
        try:
            ctx = ssl.create_default_context(cafile=str(system_certs))
            return ctx
        except Exception:
            pass

    # Try certifi
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

    return ssl.create_default_context()


def _download(url: str, description: str = "") -> str:
    """Download a URL and return its content as a string."""
    ctx = _get_ssl_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    logger.info("Downloading %s from %s ...", description, url.split("/")[-1])
    resp = urllib.request.urlopen(req, context=ctx, timeout=60)
    data = resp.read().decode("utf-8")
    logger.info("  Downloaded %d bytes", len(data))
    return data


# ── TruthfulQA Processing ────────────────────────────────────────────

def process_truthfulqa() -> pd.DataFrame:
    """
    Download TruthfulQA CSV from GitHub and create balanced
    (truthful, hallucinated) pairs.

    Each question produces:
      - 1 truthful pair:      (question, best_answer,                label=0)
      - 1 hallucinated pair:  (question, best_incorrect_answer,      label=1)

    Source CSV columns:
      Type, Category, Question, Best Answer, Best Incorrect Answer,
      Correct Answers, Incorrect Answers, Source

    Returns
    -------
    pd.DataFrame with columns: [text, label, question, answer, source]
    """
    logger.info("Processing TruthfulQA...")
    raw_csv = _download(TRUTHFULQA_URL, "TruthfulQA")

    reader = csv.DictReader(io.StringIO(raw_csv))
    rows: List[dict] = []

    for item in reader:
        question = item["Question"].strip()
        best_answer = item["Best Answer"].strip()
        best_incorrect = item.get("Best Incorrect Answer", "").strip()

        # Truthful pair
        text_truthful = f"Question: {question}\nAnswer: {best_answer}"
        rows.append({
            "text": text_truthful,
            "label": 0,
            "question": question,
            "answer": best_answer,
            "source": "truthfulqa",
        })

        # Hallucinated pair — use the "Best Incorrect Answer"
        if best_incorrect:
            text_hall = f"Question: {question}\nAnswer: {best_incorrect}"
            rows.append({
                "text": text_hall,
                "label": 1,
                "question": question,
                "answer": best_incorrect,
                "source": "truthfulqa",
            })

    df = pd.DataFrame(rows)
    logger.info(
        "  TruthfulQA processed: %d pairs (label 0: %d, label 1: %d)",
        len(df),
        (df.label == 0).sum(),
        (df.label == 1).sum(),
    )
    return df


# ── HaluEval Processing ──────────────────────────────────────────────

def process_halueval() -> pd.DataFrame:
    """
    Download HaluEval QA JSON-Lines from GitHub and create balanced
    (truthful, hallucinated) pairs.

    Each line is a JSON object with keys:
      knowledge, question, right_answer, hallucinated_answer

    Each row produces:
      - 1 truthful pair:      (question, right_answer,           label=0)
      - 1 hallucinated pair:  (question, hallucinated_answer,    label=1)

    Returns
    -------
    pd.DataFrame with columns: [text, label, question, answer, source]
    """
    logger.info("Processing HaluEval...")
    raw_json = _download(HALUEVAL_QA_URL, "HaluEval QA")

    rows: List[dict] = []
    for line in raw_json.strip().split("\n"):
        item = json.loads(line)
        question = item["question"].strip()
        right_answer = item["right_answer"].strip()
        hall_answer = item["hallucinated_answer"].strip()

        # Truthful pair
        text_truthful = f"Question: {question}\nAnswer: {right_answer}"
        rows.append({
            "text": text_truthful,
            "label": 0,
            "question": question,
            "answer": right_answer,
            "source": "halueval",
        })

        # Hallucinated pair
        text_hall = f"Question: {question}\nAnswer: {hall_answer}"
        rows.append({
            "text": text_hall,
            "label": 1,
            "question": question,
            "answer": hall_answer,
            "source": "halueval",
        })

    df = pd.DataFrame(rows)
    logger.info(
        "  HaluEval processed: %d pairs (label 0: %d, label 1: %d)",
        len(df),
        (df.label == 0).sum(),
        (df.label == 1).sum(),
    )
    return df


# ── Stratified Split ──────────────────────────────────────────────────

def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified 80/10/10 split preserving label distribution.

    Returns (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "Ratios must sum to 1.0"

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        stratify=df["label"],
    )

    # Second split: val vs test (50/50 of the remaining 20%)
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=random_seed,
        stratify=temp_df["label"],
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


# ── Save to disk ──────────────────────────────────────────────────────

def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Save train/val/test DataFrames as CSV files."""
    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = out_dir / f"{split_name}.csv"
        split_df.to_csv(path, index=False)
        logger.info("  Saved %s/%s.csv — %d samples", dataset_name, split_name, len(split_df))


# ── Summary statistics ────────────────────────────────────────────────

def print_summary(dataset_name: str, train_df, val_df, test_df) -> None:
    """Print a formatted summary table."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()} — Dataset Summary")
    print(f"{'='*60}")
    print(f"  {'Split':<10} {'Total':>8} {'Label 0':>10} {'Label 1':>10} {'Ratio':>8}")
    print(f"  {'-'*48}")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n0 = (df.label == 0).sum()
        n1 = (df.label == 1).sum()
        ratio = f"{n0/(n0+n1):.1%}/{n1/(n0+n1):.1%}"
        print(f"  {name:<10} {len(df):>8} {n0:>10} {n1:>10} {ratio:>8}")
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"  {'-'*48}")
    print(f"  {'TOTAL':<10} {total:>8}")
    print(f"{'='*60}\n")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    """Run the full data preparation pipeline."""
    print("\n" + "=" * 60)
    print("  LLM Hallucination Detection — Data Preparation")
    print("  (GitHub direct download — no HuggingFace required)")
    print("=" * 60 + "\n")

    # ── 1. Process TruthfulQA ──────────────────────────────────
    logger.info("Step 1/4: Processing TruthfulQA...")
    tqa_df = process_truthfulqa()

    logger.info("Step 2/4: Splitting TruthfulQA (80/10/10 stratified)...")
    tqa_train, tqa_val, tqa_test = stratified_split(tqa_df)
    save_splits(tqa_train, tqa_val, tqa_test, "truthfulqa")
    print_summary("TruthfulQA", tqa_train, tqa_val, tqa_test)

    # ── 2. Process HaluEval ────────────────────────────────────
    logger.info("Step 3/4: Processing HaluEval...")
    halu_df = process_halueval()

    logger.info("Step 4/4: Splitting HaluEval (80/10/10 stratified)...")
    halu_train, halu_val, halu_test = stratified_split(halu_df)
    save_splits(halu_train, halu_val, halu_test, "halueval")
    print_summary("HaluEval", halu_train, halu_val, halu_test)

    # ── 3. Quick sample inspection ─────────────────────────────
    print("\n📋 Sample data (TruthfulQA train):")
    print("-" * 50)
    for _, row in tqa_train.head(3).iterrows():
        lbl = "✅ Truthful" if row.label == 0 else "❌ Hallucinated"
        print(f"  [{lbl}] {row.text[:120]}...")
    print()

    print("📋 Sample data (HaluEval train):")
    print("-" * 50)
    for _, row in halu_train.head(3).iterrows():
        lbl = "✅ Truthful" if row.label == 0 else "❌ Hallucinated"
        print(f"  [{lbl}] {row.text[:120]}...")
    print()

    logger.info("✅ Data preparation complete! All CSVs saved to data/")
    return tqa_df, halu_df


if __name__ == "__main__":
    main()
