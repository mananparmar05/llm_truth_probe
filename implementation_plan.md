# LLM Hallucination Detection — Implementation Plan

## Background

This project implements **probing classifiers on transformer hidden states** to detect LLM hallucinations at inference time. The core insight (Azaria & Mitchell 2023): a frozen LLM's internal activations encode a **linear truthfulness signal** at mid-network layers — even when the output is confidently wrong.

The project replicates and extends that paper with:
- A **cross-dataset generalisation study** (TruthfulQA ↔ HaluEval)
- **Multi-model comparison** (Llama-3.1-8B vs. Mistral-7B)
- **Control experiment suite** (shuffled labels, random layers)
- A **production inference wrapper** (<2ms latency)

---

## Architecture

```
datasets (TruthfulQA, HaluEval)
        │
        ▼
Frozen LLM (Llama-3.1-8B fp16)
  + PyTorch Forward Hooks
        │
        ▼ [N × 32 × 4096] tensor  (~5–10 GB)
Hidden State Store (HDF5)
        │
        ▼
Probe Training (LR / MLP / SVM per layer)
  → 128 classifiers total
        │
        ▼
Analysis: Layer-wise AUROC, t-SNE, Cross-dataset, Controls
        │
        ▼
Production HallucinationDetector  →  FastAPI endpoint
```

---

## Tech Stack

| Layer | Tools | Version |
|-------|-------|---------|
| Core ML | PyTorch | 2.x |
| LLM Loading | HuggingFace Transformers | 4.40+ |
| Probe Training | scikit-learn | 1.4+ |
| Data | HuggingFace Datasets, pandas | latest |
| Visualisation | matplotlib, seaborn, plotly | latest |
| Storage | h5py (HDF5), joblib | latest |
| Tracking | W&B / MLflow | latest |
| API | FastAPI + uvicorn | latest |
| Testing | pytest + pytest-cov | latest |

**GPU requirement**: Single A100 (Google Colab Pro) or T4 with batch_size≤4

---

## Phase-by-Phase Implementation

---

### Phase 1 — Data Collection & Preparation
**Goal**: Load and preprocess TruthfulQA (817 samples) and HaluEval (10,000 samples).

#### Files
- **[MODIFY]** [data/](file:///Users/phoenix/Desktop/LLM:HALL/data/) — data will be downloaded here

**Steps**:
1. `datasets.load_dataset("truthful_qa", "generation")` — filter to binary labels
2. `datasets.load_dataset("pminervini/HaluEval")` — use correct/hallucinated answer pairs
3. Format: `"Question: {q}\nAnswer: {a}"` → binary label (0=truthful, 1=hallucinated)
4. Stratified 80/10/10 split, seed=42
5. Store as pandas DataFrame: `[text, label, source_dataset, question, answer]`

---

### Phase 2 — Model Loading & Hook Registration
**Goal**: Load frozen Llama-3.1-8B in fp16, register forward hooks on all 32 layers.

#### Files
- **[EXISTS]** [extraction/hook_extractor.py](file:///Users/phoenix/Desktop/LLM:HALL/extraction/hook_extractor.py)
- **[EXISTS]** [extraction/batch_runner.py](file:///Users/phoenix/Desktop/LLM:HALL/extraction/batch_runner.py)
- **[EXISTS]** [extraction/storage.py](file:///Users/phoenix/Desktop/LLM:HALL/extraction/storage.py)

**Steps**:
1. Load model with `torch_dtype=torch.float16, device_map="auto"` → zero gradients
2. `HiddenStateExtractor.register_hooks()` — attaches to `model.model.layers[i]`
3. Capture `output[0][:, -1, :]` — last-token hidden state at each layer
4. `BatchRunner.run()` → `all_hidden_states [N, 32, 4096]`
5. Save to HDF5 via `HiddenStateStore.save()`

**Key design decision**: Last-token position captures the richest context in causal attention.

---

### Phase 3 — Probe Classifier Training
**Goal**: Train 4 probe types × 32 layers = 128 classifiers.

#### Files
- **[EXISTS]** [probing/probe_trainer.py](file:///Users/phoenix/Desktop/LLM:HALL/probing/probe_trainer.py)
- **[EXISTS]** [probing/probe_evaluator.py](file:///Users/phoenix/Desktop/LLM:HALL/probing/probe_evaluator.py)
- **[EXISTS]** [probing/probe_selector.py](file:///Users/phoenix/Desktop/LLM:HALL/probing/probe_selector.py)

**Probe types**:
| Type | Class | Why |
|------|-------|-----|
| Logistic Regression | `LogisticRegression` | Linear separability test |
| MLP | `MLPClassifier(256, 64)` | Non-linear capacity |
| SVM (RBF) | `SVC(probability=True)` | Margin-based, robust |
| Ensemble | Stacked on top-5 layers | Highest ceiling |

**Expected finding**: AUROC rises from ~0.51 at layer 1 → ~0.96 at layers 14–16 → drops to ~0.80 at layer 32.

---

### Phase 4 — Visualisation & Analysis
**Goal**: t-SNE scatter plot, layer-wise AUROC bar chart, ROC curve.

#### Files
- **[EXISTS]** [analysis/tsne_visualizer.py](file:///Users/phoenix/Desktop/LLM:HALL/analysis/tsne_visualizer.py)
- **[EXISTS]** [analysis/layer_heatmap.py](file:///Users/phoenix/Desktop/LLM:HALL/analysis/layer_heatmap.py)

**Key outputs**:
- `results/plots/tsne_layer15.png` — **money shot** for README/portfolio
- `results/plots/auroc_bar_llama_truthfulqa.png` — shows peak at layers 14–16
- `results/plots/model_comparison_heatmap.png` — Llama vs. Mistral

---

### Phase 5 — Cross-Dataset Generalisation Study ⭐ (Original Contribution)
**Goal**: 4-experiment study: train on A, test on B (and vice versa).

#### Files
- **[EXISTS]** [analysis/cross_dataset.py](file:///Users/phoenix/Desktop/LLM:HALL/analysis/cross_dataset.py)

| Experiment | Expected AUROC |
|------------|---------------|
| TruthfulQA → TruthfulQA | ~0.94–0.96 (baseline) |
| HaluEval → HaluEval | ~0.90–0.94 (baseline) |
| TruthfulQA → HaluEval | ~0.85–0.90 (**original**) |
| HaluEval → TruthfulQA | ~0.82–0.88 (**original**) |

**Research significance**: AUROC >0.85 cross-dataset → probe learns a **universal truthfulness direction**, not benchmark-specific artifacts.

---

### Phase 6 — Control Experiments (Null Hypothesis Testing)
**Goal**: Prove the probe doesn't pick up spurious patterns.

#### Files
- **[EXISTS]** [analysis/control_experiments.py](file:///Users/phoenix/Desktop/LLM:HALL/analysis/control_experiments.py)

| Control | Method | Expected AUROC |
|---------|--------|---------------|
| Shuffled labels | Same hidden states, random label permutation | ~0.50 |
| Random layer | Train probe on layer 1 (token embeddings) | ~0.52 |
| Random baseline | Diagonal on ROC plot | 0.50 |

---

### Phase 7 — Production Inference Pipeline
**Goal**: Wrap any HuggingFace LLM with the trained probe, <2ms latency.

#### Files
- **[EXISTS]** [inference/realtime_detector.py](file:///Users/phoenix/Desktop/LLM:HALL/inference/realtime_detector.py)
- **[EXISTS]** [inference/api_wrapper.py](file:///Users/phoenix/Desktop/LLM:HALL/inference/api_wrapper.py)

**Usage**:
```python
detector = HallucinationDetector.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B",
    probe_path="results/models/probe_layer14_logistic_regression.pkl",
)
result = detector.detect(question, answer)
# → {"hallucination_probability": 0.92, "verdict": "🚫 HALLUCINATION"}
```

---

## Key Metrics Targets

| Metric | Target |
|--------|--------|
| Best AUROC (TruthfulQA) | ≥0.94 |
| Best AUROC (HaluEval) | ≥0.90 |
| Cross-dataset AUROC | ≥0.85 |
| Shuffled-label AUROC | ≤0.55 |
| Probe training time/layer | <60s CPU |
| Inference latency | <2ms |

---

## Verification Plan

### Automated Tests
```bash
# Run full test suite (no GPU required — uses synthetic 64-dim hidden states)
pytest tests/test_pipeline.py -v --tb=short --cov=. --cov-report=term-missing

# Smoke test dependencies
python3 -c "import torch, transformers, sklearn; print('OK')"
```

### Manual Verification (GPU required)
1. Run `notebooks/02_hidden_state_extraction.ipynb` — verify tensor shape `[N, 32, 4096]`
2. Run `notebooks/03_probe_training.ipynb` — check AUROC peak at layers 14–16
3. Inspect `results/plots/tsne_layer15.png` — two clean clusters = success
4. Hit the FastAPI endpoint: `POST /detect` with a known hallucinated answer
