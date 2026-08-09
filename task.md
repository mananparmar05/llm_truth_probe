# LLM Hallucination Detection — Project Task Sequence

## Phase 0 — Environment Setup & Scaffold (COMPLETE ✅)
- [x] Read PDF and extract full project specification
- [x] Create project directory structure (`data/`, `extraction/`, `probing/`, `analysis/`, `inference/`, `notebooks/`, `tests/`, `results/`, `configs/`)
- [x] Write pinned `requirements.txt`
- [x] Write `configs/config.yaml` with model, dataset, probing, and API settings
- [x] Write `setup_env.sh` environment bootstrap script
- [x] Write `.gitignore` excluding large models, tensors, and cache files
- [x] Write comprehensive `README.md` with metrics, architecture, and instructions
- [x] Set up Python 3.10 `.venv` environment with PyTorch 2.2.2 + NumPy 1.26.4 fix
- [x] Install all dependencies (transformers, datasets, scikit-learn, h5py, fastapi, pytest, etc.)
- [x] Create core modules:
  - `extraction/hook_extractor.py` (PyTorch forward hook capture)
  - `extraction/batch_runner.py` (batched extraction runner)
  - `extraction/storage.py` (HDF5 & numpy tensor store)
  - `probing/probe_trainer.py` (LR, MLP, SVM, Ensemble layer trainer)
  - `probing/probe_evaluator.py` (AUROC, F1, ROC, CM evaluator)
  - `probing/probe_selector.py` (best layer & probe picker)
  - `analysis/tsne_visualizer.py` (PCA → t-SNE 2D scatter plot)
  - `analysis/layer_heatmap.py` (layer-wise AUROC bar chart & heatmap)
  - `analysis/cross_dataset.py` (4-experiment train→test study)
  - `analysis/control_experiments.py` (shuffled-label & random-layer controls)
  - `inference/realtime_detector.py` (production detector wrapper class)
  - `inference/api_wrapper.py` (FastAPI serving endpoint)
- [x] Write unit test suite in `tests/test_pipeline.py`
- [x] Create Jupyter Notebook stubs (`notebooks/01_data_exploration.ipynb` through `05_cross_dataset_generalization.ipynb`)
- [x] Initialize Git repository & create initial commit (`285ba48`)

---

## Phase 1 — Data Collection & Preparation
- [ ] Open terminal & activate virtualenv: `source .venv/bin/activate`
- [ ] Run `notebooks/01_data_exploration.ipynb`
- [ ] Load TruthfulQA dataset: `datasets.load_dataset("truthful_qa", "generation")`
- [ ] Load HaluEval dataset: `datasets.load_dataset("pminervini/HaluEval")`
- [ ] Preprocess & format samples: `"Question: {q}\nAnswer: {a}"` with binary label (0=Truthful, 1=Hallucinated)
- [ ] Perform stratified 80/10/10 split (random_state=42)
- [ ] Save processed DataFrames to `data/truthfulqa/` and `data/halueval/`
- [ ] Verify dataset counts: 817 TruthfulQA pairs, 10,000 HaluEval pairs

---

## Phase 2 — Model Loading & Hidden State Extraction (GPU / Colab / Local)
- [ ] Open `notebooks/02_hidden_state_extraction.ipynb`
- [ ] Authenticate HuggingFace Hub: `huggingface-cli login`
- [ ] Load target LLM: `meta-llama/Llama-3.1-8B` (fp16, frozen parameters)
- [ ] Attach `HiddenStateExtractor` hooks to all 32 transformer layers
- [ ] Run `BatchRunner` on TruthfulQA → extract shape `[817, 32, 4096]`
- [ ] Run `BatchRunner` on HaluEval → extract shape `[10000, 32, 4096]`
- [ ] Save hidden-state tensors to HDF5 store: `data/hidden_states/hidden_states.h5`
- [ ] (Optional) Repeat extraction for secondary model `mistralai/Mistral-7B-v0.1`

---

## Phase 3 — Probe Classifier Training (CPU)
- [ ] Open `notebooks/03_probe_training.ipynb`
- [ ] Load extracted hidden states from `data/hidden_states/hidden_states.h5`
- [ ] Instantiate `ProbeTrainer` with probe types: `LogisticRegression`, `MLP`, `SVM`
- [ ] Train 128 probes across all 32 layers (`trainer.train_all_layers()`)
- [ ] Train top-5 layer stacking ensemble (`trainer.build_ensemble()`)
- [ ] Find best layer & probe type using `ProbeSelector` (expected peak at Layer 14–16, AUROC ~0.94–0.96)
- [ ] Save trained probes to `results/models/` using `joblib`

---

## Phase 4 — Visualisation & Interpretability
- [ ] Open `notebooks/04_results_analysis.ipynb`
- [ ] Generate t-SNE scatter plot for Layer 15 → save to `results/plots/tsne_layer15.png`
- [ ] Generate layer-wise AUROC bar chart → save to `results/plots/auroc_bar_llama_truthfulqa.png`
- [ ] Generate ROC curve & confusion matrix for best probe → save to `results/plots/roc_best_probe.png`
- [ ] Generate multi-model comparison heatmap (Llama-3.1-8B vs. Mistral-7B)

---

## Phase 5 — Cross-Dataset Generalisation Study ⭐
- [ ] Open `notebooks/05_cross_dataset_generalization.ipynb`
- [ ] Execute 4 train → test generalisation experiments:
  1. TruthfulQA → TruthfulQA (baseline)
  2. HaluEval → HaluEval (baseline)
  3. TruthfulQA → HaluEval (zero-shot cross-dataset evaluation)
  4. HaluEval → TruthfulQA (zero-shot cross-dataset evaluation)
- [ ] Plot cross-dataset layer curves → save to `results/plots/cross_dataset_auroc.png`
- [ ] Confirm cross-dataset AUROC retention > 0.85

---

## Phase 6 — Control Experiments (Null Hypothesis Testing)
- [ ] Run `ControlExperiments.shuffled_label_control()` → confirm AUROC ≈ 0.50
- [ ] Run `ControlExperiments.random_layer_control()` → confirm Layer 1 AUROC ≈ 0.52
- [ ] Generate control comparison figure → save to `results/plots/control_comparison.png`

---

## Phase 7 — Production Inference & API Serving
- [ ] Load best saved probe (`results/models/probe_layer14_logistic_regression.pkl`)
- [ ] Instantiate `HallucinationDetector` wrapper class
- [ ] Run live inference test on sample QA pairs
- [ ] Launch FastAPI server locally: `uvicorn inference.api_wrapper:app --host 0.0.0.0 --port 8000`
- [ ] Verify `POST /detect` endpoint response time (<2ms probe latency overhead)

---

## Phase 8 — Local Verification Commands
```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run unit test suite
pytest tests/test_pipeline.py -v

# 3. Check core module imports
python -c "import torch, transformers, datasets, sklearn, h5py; print('All core modules OK ✅')"
```
