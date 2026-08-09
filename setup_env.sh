#!/usr/bin/env bash
# =============================================================
# setup_env.sh  —  One-shot environment bootstrap for
#                  LLM Hallucination Detection project
# =============================================================
# Usage:
#   bash setup_env.sh            # creates/updates virtual env
#   bash setup_env.sh --conda    # use conda instead of venv
# =============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
CONDA_ENV="llm-hall"
PYTHON_MIN="3.10"

log()  { echo "[setup] $*"; }
warn() { echo "[WARN]  $*" >&2; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

# ── Parse flags ───────────────────────────────────────────────
USE_CONDA=false
for arg in "$@"; do
  [[ "$arg" == "--conda" ]] && USE_CONDA=true
done

# ── Check Python version ──────────────────────────────────────
check_python() {
  local py="$1"
  local ver
  ver="$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" 2>/dev/null || return 1
  python3 -c "
v = '$ver'.split('.')
req = '$PYTHON_MIN'.split('.')
assert int(v[0]) > int(req[0]) or (int(v[0]) == int(req[0]) and int(v[1]) >= int(req[1])), 'version too old'
" 2>/dev/null
}

# ── CONDA path ────────────────────────────────────────────────
setup_conda() {
  log "Setting up conda environment: $CONDA_ENV"
  command -v conda >/dev/null 2>&1 || die "conda not found. Install Miniconda first."

  if conda env list | grep -q "^$CONDA_ENV "; then
    log "Conda env '$CONDA_ENV' already exists — updating packages."
  else
    log "Creating conda env '$CONDA_ENV' with Python ${PYTHON_MIN}+"
    conda create -y -n "$CONDA_ENV" python="${PYTHON_MIN}" pip
  fi

  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
  install_deps
  log ""
  log "✅  Done! Activate with:  conda activate $CONDA_ENV"
}

# ── VENV path ─────────────────────────────────────────────────
setup_venv() {
  log "Setting up virtual environment at $VENV_DIR"

  # Find a suitable Python
  PYTHON_BIN=""
  for py in python3.12 python3.11 python3.10 python3; do
    if command -v "$py" >/dev/null 2>&1 && check_python "$py"; then
      PYTHON_BIN="$py"
      break
    fi
  done
  [[ -z "$PYTHON_BIN" ]] && die "Python ${PYTHON_MIN}+ not found. Install it first."
  log "Using Python: $PYTHON_BIN ($("$PYTHON_BIN" --version))"

  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    log "Virtual environment created."
  else
    log "Virtual environment already exists — updating."
  fi

  # Activate
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  install_deps
  log ""
  log "✅  Done! Activate with:  source $VENV_DIR/bin/activate"
}

# ── Shared install ─────────────────────────────────────────────
install_deps() {
  log "Upgrading pip / wheel / setuptools …"
  pip install --quiet --upgrade pip wheel setuptools

  # ── PyTorch: try PyPI first, then PyTorch nightly (for Python 3.13) ──
  log "Installing PyTorch …"
  if ! pip install --quiet torch torchvision 2>/dev/null; then
    log "PyPI torch not found for this Python version — trying PyTorch nightly index …"
    if ! pip install --quiet torch torchvision \
        --index-url https://download.pytorch.org/whl/nightly/cpu 2>/dev/null; then
      warn "Could not install PyTorch automatically."
      warn "On macOS (Python 3.13), install manually:"
      warn "  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu"
      warn "Or use Google Colab for GPU-backed extraction."
    fi
  fi

  # ── Non-PyTorch deps (always installable) ─────────────────────
  log "Installing core project dependencies …"
  pip install --quiet \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    datasets>=2.18.0 \
    huggingface_hub>=0.22.0 \
    pandas>=2.0.0 \
    numpy>=1.26.0 \
    scikit-learn>=1.4.0 \
    joblib>=1.3.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    plotly>=5.20.0 \
    h5py>=3.10.0 \
    safetensors>=0.4.2 \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    pydantic>=2.6.0 \
    jupyterlab>=4.1.0 \
    ipywidgets>=8.1.0 \
    tqdm>=4.66.0 \
    black>=24.3.0 \
    flake8>=7.0.0 \
    pytest>=8.1.0 \
    pytest-cov>=5.0.0 \
    python-dotenv>=1.0.0 \
    mlflow>=2.11.0 \
    wandb>=0.16.0

  # Install NLTK data (if needed)
  python3 -c "import nltk; nltk.download('punkt', quiet=True)" 2>/dev/null || true

  log "All dependencies installed."
}

# ── Set PYTHONPATH ────────────────────────────────────────────
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ── Main ──────────────────────────────────────────────────────
cd "$PROJECT_DIR"

if $USE_CONDA; then
  setup_conda
else
  setup_venv
fi

# ── Quick smoke test ──────────────────────────────────────────
log "Running smoke tests (no GPU required) …"
python3 -c "
import numpy as np
import sklearn
import torch
import transformers
print(f'  numpy      : {np.__version__}')
print(f'  scikit-learn: {sklearn.__version__}')
print(f'  torch      : {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print('  Smoke test PASSED ✅')
"
