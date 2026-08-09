# LLM Hallucination Detection via Probing Classifiers

> **Can we detect, from inside a model's own internal representations — before output is shown — whether it is about to hallucinate?**
>
> This project answers: **yes**, with AUROC up to **0.96** using a logistic probe on Layer 15 hidden states of a frozen Llama-3.1-8B.

---

## 🧠 What This Project Does

1. Loads **TruthfulQA** (817 questions) and **HaluEval** (10,000 QA pairs) benchmark datasets
2. Runs every sample through a **frozen Llama-3.1-8B** (or Mistral-7B) via PyTorch forward hooks
3. Captures the **last-token hidden state** at all 32 transformer layers → `[N × 32 × 4096]` tensor
4. Trains **128 lightweight probes** (logistic regression, MLP, SVM, ensemble) — one per layer per type
5. Finds **Layer 14–16** as the peak truthfulness signal zone
6. Runs a **cross-dataset generalisation study** and **control experiments**
7. Wraps the best probe in a **production inference class** adding <2ms latency

---

## 📁 Project Structure

```
LLM:HALL/
├── data/
│   ├── truthfulqa/          ← TruthfulQA (loaded from HF Hub)
│   ├── halueval/            ← HaluEval (loaded from HF Hub)
│   └── hidden_states/       ← Extracted hidden state tensors (.h5)
│
├── extraction/
│   ├── hook_extractor.py    ← PyTorch forward hook registration
│   ├── batch_runner.py      ← Batched forward passes
│   └── storage.py           ← HDF5 / numpy persistence
│
├── probing/
│   ├── probe_trainer.py     ← Train LR / MLP / SVM per layer
│   ├── probe_evaluator.py   ← AUROC, F1, ROC, cross-dataset eval
│   └── probe_selector.py    ← Best layer + probe type selection
│
├── analysis/
│   ├── tsne_visualizer.py   ← t-SNE / PCA scatter plots
│   ├── layer_heatmap.py     ← AUROC bar charts + heatmaps
│   ├── cross_dataset.py     ← 4-experiment generalisation study
│   └── control_experiments.py ← Shuffled-label & random-layer controls
│
├── inference/
│   ├── realtime_detector.py ← HallucinationDetector class
│   └── api_wrapper.py       ← FastAPI serving endpoint
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_hidden_state_extraction.ipynb
│   ├── 03_probe_training.ipynb
│   ├── 04_results_analysis.ipynb
│   └── 05_cross_dataset_generalization.ipynb
│
├── tests/
│   └── test_pipeline.py
│
├── configs/
│   └── config.yaml
│
├── requirements.txt
└── setup_env.sh
```

---

## ⚙️ Multi-Platform System Setup Guide

This guide covers setting up the project on **any system** (Linux CUDA GPU server, macOS, Windows WSL2, Conda, or Google Colab).

### System Prerequisites
- **Python**: `3.10` or `3.11` (Recommended for PyTorch 2.x & CUDA compatibility)
- **RAM**: Minimum 16 GB (32 GB recommended)
- **GPU (Recommended for Hidden State Extraction)**: NVIDIA GPU with 16GB+ VRAM (A100, H100, RTX 3090/4090, or T4). *Probe training and evaluation run fast on CPU.*

---

### Option A: Automated One-Shot Setup (Linux / macOS / WSL2)

```bash
# 1. Clone repository
git clone https://github.com/mananparmar05/llm_truth_probe.git
cd llm_truth_probe

# 2. Run automated environment setup
bash setup_env.sh

# 3. Activate virtual environment
source .venv/bin/activate
```

---

### Option B: Manual Virtualenv Setup (Linux / macOS / Windows WSL2)

```bash
# 1. Create a Python 3.10 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate   # On Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Upgrade core tooling
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch according to your Hardware:

# ---> A) CUDA 12.1 (Linux / Windows CUDA GPU Server)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ---> B) CUDA 11.8 (Older GPU Drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ---> C) CPU / Apple Silicon MPS (macOS M1/M2/M3/M4 or CPU-only server)
pip install torch torchvision

# 4. Install NumPy 1.x (to avoid PyTorch 2.x NumPy 2.0 C-extension warning)
pip install "numpy<2.0"

# 5. Install all project dependencies
pip install -r requirements.txt
```

---

### Option C: Conda / Mamba Setup

```bash
# 1. Create conda environment with Python 3.10
conda create -n llm-hall python=3.10 -y
conda activate llm-hall

# 2. Install PyTorch with CUDA support (or cpu/mps for macOS)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. Install project dependencies
pip install "numpy<2.0"
pip install -r requirements.txt
```

---

### Option D: Google Colab / Kaggle (Free GPU A100 / T4)

Create a new Colab Notebook and run the first cell:

```python
# Colab Setup Cell
!git clone https://github.com/mananparmar05/llm_truth_probe.git
%cd llm_truth_probe
!pip install -q -r requirements.txt "numpy<2.0"
!huggingface-cli login
```

---

## 🚀 Quick Execution Guide

Once your environment is set up and activated:

```bash
# 1. Run unit test suite
pytest tests/test_pipeline.py -v

# 2. Verify all core modules import cleanly
python -c "import torch, transformers, datasets, sklearn, h5py; print('All core modules OK ✅')"

# 3. Open Jupyter Lab for interactive notebooks
jupyter lab
```

### Running the Full Pipeline

```bash
# Phase 1+2: Extract hidden states (Requires GPU)
python -m extraction.batch_runner

# Phase 3: Train probes across 32 layers (CPU)
python -m probing.probe_trainer

# Phase 4: Generate t-SNE & AUROC plots
python -m analysis.tsne_visualizer
python -m analysis.layer_heatmap

# Phase 5: Run cross-dataset generalisation study
python -m analysis.cross_dataset

# Phase 6: Run control experiments
python -m analysis.control_experiments

# Phase 7: Launch live detection API server
uvicorn inference.api_wrapper:app --host 0.0.0.0 --port 8000
```

---

## 📊 Key Results

| Metric | Value | Context |
|--------|-------|---------|
| Best AUROC (LR, Layer 15, TruthfulQA) | ~0.94–0.96 | Replication of Azaria & Mitchell (2023) |
| Best AUROC (MLP, Layer 15) | ~0.95–0.97 | Small gain from non-linearity |
| Cross-dataset AUROC (TruthfulQA→HaluEval) | ~0.85–0.90 | **Original contribution** |
| Shuffled-label control AUROC | ~0.50 | Confirms no memorisation |
| Random-layer control (Layer 1) | ~0.52 | Confirms layer-specificity |
| **Peak signal zone** | **Layers 14–16 of 32** | Mid-network, consistent with literature |
| Probe training time (per layer) | <60s on CPU | Extremely lightweight |
| Inference overhead | <2ms per query | Negligible in production |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Core ML | PyTorch 2.x, HuggingFace Transformers 4.40+ |
| Probe Training | scikit-learn 1.4+ |
| Data | HuggingFace Datasets, pandas |
| Visualisation | matplotlib, seaborn, plotly |
| Storage | HDF5 (h5py), joblib |
| Experiment Tracking | W&B / MLflow |
| API | FastAPI + uvicorn |
| Testing | pytest + pytest-cov |

---

## 📚 References

- Azaria, A. & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying.* EMNLP Findings.
- Lin, S. et al. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* ACL.
- Li, J. et al. (2023). *HaluEval: A Large-Scale Hallucination Evaluation Benchmark.* EMNLP.
