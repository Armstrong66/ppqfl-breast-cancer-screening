#!/usr/bin/env bash
# =============================================================================
# setup_local.sh — Local Machine Setup for QML Pipeline (RTX workstation)
# =============================================================================
# Run this ONCE after copying your scripts from Kaggle to the local machine.
# It handles:
#   1. Auto-detecting and rewriting Kaggle paths to local paths
#   2. Setting up a Python virtual environment
#   3. Installing all dependencies from requirements.txt
#   4. Verifying GPU availability
#   5. Printing nohup commands for overnight runs
#
# Usage (MobaXTerm or any Linux terminal):
#   chmod +x setup_local.sh
#   ./setup_local.sh
#   ./setup_local.sh --data-dir /path/to/your/data
# =============================================================================

set -euo pipefail

# ── Defaults — edit these to match your machine ──────────────────────────────

# Directory where all pipeline scripts live (where this file is)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Where you want pipeline outputs to be written
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

# Where your dataset files live (the two datasets)
# Expected sub-structure:
#   DATA_DIR/mendeley/Breast Cancer Dataset/Breast Cancer Original/Benign/
#   DATA_DIR/mendeley/Breast Cancer Dataset/Breast Cancer Original/Malignant/
#   DATA_DIR/kau/BIRAD1/b1/  etc.
# Default: looks for dataset relative to script location first
# You can override with: ./setup_local.sh --data-dir /path/to/your/data
SCRIPT_DIR_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-}"

# Auto-detect path if not provided
if [[ -z "$DATA_DIR" ]] || [[ ! -d "$DATA_DIR/mendeley" && ! -d "$DATA_DIR/kau" ]]; then
    # Check if datasets exist relative to script location
    if [[ -d "$SCRIPT_DIR_ABS/Breast Cancer Dataset" ]]; then
        DATA_DIR="$SCRIPT_DIR_ABS"
    elif [[ -d "$SCRIPT_DIR_ABS/../Breast Cancer Dataset" ]]; then
        DATA_DIR="$SCRIPT_DIR_ABS/.."
    elif [[ -d "/data/derrick/mendeley/Breast Cancer Dataset/Breast Cancer Original" ]]; then
        # Server path
        DATA_DIR="/data/derrick"
    else
        echo "ERROR: Cannot locate dataset directories."
        echo "Please provide the data directory with: ./setup_local.sh --data-dir /path/to/data"
        exit 1
    fi
fi

# Python executable — change to python3.12 or full path if needed
PYTHON="python3"

# Virtual environment name
VENV_DIR="${SCRIPT_DIR}/.venv"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2";   shift 2 ;;
        --python)     PYTHON="$2";     shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Colour helpers
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
info() { echo -e "  $1"; }
head() { echo -e "\n${BOLD}${BLUE}── $1 ──${NC}"; }

echo -e "${BOLD}${BLUE}"
echo "═══════════════════════════════════════════════════════════════"
echo "  QML Pipeline — Local Machine Setup"
echo "  Script dir : $SCRIPT_DIR"
echo "  Output dir : $OUTPUT_DIR"
echo "  Data dir   : $DATA_DIR"
echo "═══════════════════════════════════════════════════════════════"
echo -e "${NC}"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  PATH REWRITING
# Replaces all Kaggle-specific paths in .py scripts with local equivalents.
# Safe: writes to a backup first, only touches Python files.
# ══════════════════════════════════════════════════════════════════════════════
head "Step 1: Rewriting Kaggle paths → local paths"

# Back up originals before touching anything
BACKUP_DIR="${SCRIPT_DIR}/.path_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp "${SCRIPT_DIR}"/*.py "$BACKUP_DIR/" 2>/dev/null || true
info "Originals backed up to: $BACKUP_DIR"

# Build the Mendeley full path from DATA_DIR
MENDELEY_ORIGINAL="${DATA_DIR}/mendeley/Breast Cancer Dataset/Breast Cancer Original"
KAU_ROOT="${DATA_DIR}/kau"

# Run the path rewriter
$PYTHON - <<PYEOF
import pathlib, re, sys

script_dir   = pathlib.Path("${SCRIPT_DIR}")
output_dir   = pathlib.Path("${OUTPUT_DIR}")
mendeley_dir = pathlib.Path("${MENDELEY_ORIGINAL}")
kau_dir      = pathlib.Path("${KAU_ROOT}")

# Map: (old_string, new_string)
replacements = [
    # Output / working directory
    ("/kaggle/working",  str(output_dir)),
    # Mendeley dataset
    (
        "/kaggle/input/datasets/josephderrick/mendeley-mammogram-image-dataset/"
        "Mammogram Image Dataset for Breast Cancer Detectio/Breast Cancer Dataset/"
        "Breast Cancer Original",
        str(mendeley_dir)
    ),
    # KAU dataset root
    (
        "/kaggle/input/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset",
        str(kau_dir)
    ),
]

py_files = sorted(script_dir.glob("*.py"))
changed  = 0
for pf in py_files:
    txt = pf.read_text(encoding="utf-8")
    new = txt
    for old, rep in replacements:
        new = new.replace(old, rep)
    if new != txt:
        pf.write_text(new, encoding="utf-8")
        print(f"  Updated: {pf.name}")
        changed += 1

print(f"  {changed}/{len(py_files)} scripts updated")
if changed == 0:
    print("  (Paths may already be local, or DATA_DIR/OUTPUT_DIR need adjusting)")
PYEOF

ok "Path rewriting complete"

# ══════════════════════════════════════════════════════════════════════════════
# 2.  CREATE OUTPUT DIRECTORY STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
head "Step 2: Creating output directories"

for subdir in eda_outputs baseline_outputs feature_outputs \
              vqc_outputs/regime_A vqc_outputs/regime_B \
              vqc_outputs/sweep vqc_outputs/noise \
              uq_outputs qfl_outputs external_val_outputs pipeline_logs; do
    mkdir -p "${OUTPUT_DIR}/${subdir}"
done
ok "Output directories created under: $OUTPUT_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# 3.  PYTHON VIRTUAL ENVIRONMENT
#     COMMENTED OUT — you already have a conda environment for this project.
#     Activate your conda env before running this script:
#         conda activate <your_env_name>
#     The venv block below is preserved for reference if needed on a machine
#     without conda, or for a clean pip-only setup in the future.
# ══════════════════════════════════════════════════════════════════════════════
# head "Step 3: Python virtual environment"
#
# if [[ ! -d "$VENV_DIR" ]]; then
#     info "Creating venv at $VENV_DIR..."
#     $PYTHON -m venv "$VENV_DIR"
#     ok "Virtual environment created"
# else
#     info "Venv already exists: $VENV_DIR"
# fi
#
# # Activate
# source "${VENV_DIR}/bin/activate"
# PYTHON="${VENV_DIR}/bin/python"
# PIP="${VENV_DIR}/bin/pip"
# ok "Activated: $VENV_DIR"

# Use whatever python/pip is active in the current environment (your conda env)
PIP="$PYTHON -m pip"

head "Step 3: Using active environment"
info "Python: $($PYTHON --version)"
info "Location: $(which $PYTHON)"
info "(conda env — no venv created)"

# ══════════════════════════════════════════════════════════════════════════════
# 4.  INSTALL DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════════════
head "Step 4: Installing dependencies"

$PYTHON -m pip install --upgrade pip

# ── PyTorch: use conda's existing torch if already installed ─────────────────
# Checking if torch is already present avoids a multi-GB re-download.
# PyTorch installed via conda already links correctly to the system CUDA driver
# without needing a matching nvcc/toolkit version — the driver compatibility
# is handled by conda's cudatoolkit package, not nvcc.
# Only installs via pip if torch is genuinely missing from the active env.

if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)")
    GPU=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))")
    ok "PyTorch ${TORCH_VER} already installed with CUDA — GPU: ${GPU}"
    info "Skipping PyTorch download (already present in conda env)"
    info "If you need to upgrade: conda update pytorch torchvision -c pytorch -c nvidia"
else
    info "PyTorch not found or CUDA unavailable in current env."
    info "Installing via pip with progress output..."
    info ""
    info "NOTE: If this is slow or fails, the recommended approach is:"
    info "  conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia"
    info "  (conda resolves CUDA compatibility automatically, no nvcc match needed)"
    info ""
    # Install without -q so download progress is visible
    # --no-cache-dir forces re-download if a corrupt cache is causing hangs
    $PYTHON -m pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu124 \
        --no-cache-dir \
        --progress-bar on
fi

# ── Non-PyTorch dependencies (fast, show progress) ───────────────────────────
info "Installing remaining dependencies (pennylane, flwr, sklearn, etc.)..."
$PYTHON -m pip install \
    pennylane pennylane-lightning \
    flwr \
    numpy pandas scipy scikit-learn \
    matplotlib seaborn Pillow tqdm joblib \
    --progress-bar on

ok "All dependencies installed"

# ══════════════════════════════════════════════════════════════════════════════
# 5.  VERIFY GPU + ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════
head "Step 5: Environment verification"

$PYTHON - <<PYEOF
import torch, pennylane, flwr, sklearn, torchvision
print(f"  Python     : {__import__('sys').version.split()[0]}")
print(f"  PyTorch    : {torch.__version__}")
print(f"  CUDA avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  PennyLane  : {pennylane.__version__}")
print(f"  Flower     : {flwr.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  torchvision: {torchvision.__version__}")
PYEOF

ok "Environment verified"

# ══════════════════════════════════════════════════════════════════════════════
# 6.  GENERATE nohup RUNNER COMMANDS
# ══════════════════════════════════════════════════════════════════════════════
head "Step 6: nohup commands for overnight runs"

# Resolve the active python path once (works for conda or any env)
ACTIVE_PYTHON="$(which python3)"
RUNNER="${SCRIPT_DIR}/run_pipeline.sh"
CONDA_ENV_NAME="${CONDA_DEFAULT_ENV:-your_env_name}"

cat << CMDS

  ┌─────────────────────────────────────────────────────────────────────┐
  │  READY TO RUN — copy-paste these commands into MobaXTerm            │
  └─────────────────────────────────────────────────────────────────────┘

  ── Activate your conda environment (do this in every new terminal) ──
  conda activate ${CONDA_ENV_NAME}
  cd ${SCRIPT_DIR}

  ── Run full pipeline overnight (detached, survives terminal close) ──
  mkdir -p pipeline_logs && nohup bash ${RUNNER} > pipeline_logs/nohup_full.log 2>&1 &
  echo "PID: \$!"

  ── Monitor progress live (in a second terminal tab) ──
  tail -f pipeline_logs/nohup_full.log

  ── Resume from a specific stage (after any interruption) ──
  mkdir -p pipeline_logs && nohup bash ${RUNNER} --from week3_5_vqc > pipeline_logs/nohup_resume.log 2>&1 &

  ── Run one stage only ──
  mkdir -p pipeline_logs && nohup bash ${RUNNER} --only week8_9_qfl > pipeline_logs/nohup_qfl.log 2>&1 &

  ── Run a single script directly (most flexible, easiest to debug) ──
  mkdir -p pipeline_logs && nohup ${ACTIVE_PYTHON} week3_5_vqc.py > pipeline_logs/week3_5_vqc.log 2>&1 &
  tail -f pipeline_logs/week3_5_vqc.log    # watch it live in another tab

  ── Check if a job is still running ──
  ps aux | grep python | grep -v grep

  ── Kill a running job gracefully ──
  kill <PID>       # sends SIGTERM — lets Python save checkpoint first
  kill -9 <PID>    # hard kill if SIGTERM is ignored

  ── Screen (recommended — reattach from any session, even after disconnect) ──
  screen -S qml_pipeline
      conda activate ${CONDA_ENV_NAME}
      cd ${SCRIPT_DIR}
      bash ${RUNNER}
  # Detach (keep running): Ctrl+A then D
  # List sessions:         screen -ls
  # Reattach:              screen -r qml_pipeline

CMDS

# Write the commands to a file for easy reference
CMDS_FILE="${SCRIPT_DIR}/run_commands.txt"
cat > "$CMDS_FILE" << CMDSFILE
# QML Pipeline — Run Commands
# Generated: $(date)
# Script dir: ${SCRIPT_DIR}
# Conda env:  ${CONDA_DEFAULT_ENV:-your_env_name}

# Activate conda env (do this in every new terminal)
conda activate ${CONDA_DEFAULT_ENV:-your_env_name}
cd ${SCRIPT_DIR}

# Full pipeline (nohup — survives MobaXTerm disconnect)
mkdir -p pipeline_logs && nohup bash run_pipeline.sh > pipeline_logs/nohup_full.log 2>&1 &
echo "PID: \$!"

# Monitor live
tail -f pipeline_logs/nohup_full.log

# Resume from stage
mkdir -p pipeline_logs && nohup bash run_pipeline.sh --from week3_5_vqc > pipeline_logs/nohup_resume.log 2>&1 &

# Single stage
mkdir -p pipeline_logs && nohup bash run_pipeline.sh --only week8_9_qfl > pipeline_logs/nohup_qfl.log 2>&1 &

# Single script (most direct)
mkdir -p pipeline_logs && nohup python3 week3_5_vqc.py > pipeline_logs/week3_5_vqc.log 2>&1 &
tail -f pipeline_logs/week3_5_vqc.log

# Screen session (best for long runs — reattach after disconnect)
screen -S qml_pipeline
# then inside screen: conda activate ${CONDA_DEFAULT_ENV:-your_env_name} && bash run_pipeline.sh
# Ctrl+A D to detach; screen -r qml_pipeline to reattach
CMDSFILE

ok "Run commands saved to: $CMDS_FILE"

echo ""
echo -e "${BOLD}${GREEN}  Setup complete. See run_commands.txt for copy-paste commands.${NC}"
echo ""