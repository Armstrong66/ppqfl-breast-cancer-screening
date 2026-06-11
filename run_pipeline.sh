#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Full QML Pipeline Orchestrator
# =============================================================================
# Project : Privacy-Preserving Quantum Federated Learning for Breast Cancer
#           Screening in African and MENA Populations
#
# Usage:
#   chmod +x run_pipeline.sh
#   mkdir -p pipeline_logs && nohup bash run_pipeline.sh > pipeline_logs/nohup_full.log 2>&1 & echo "PID: $!"
#
#   ./run_pipeline.sh --from 3_5_vqc   # resume from stage
#   ./run_pipeline.sh --only 8_9_qfl   # run one stage only
#   ./run_pipeline.sh --skip-slow      # minimal VQC sweep (full by default)
#
# Stage names (use exactly these strings with --from / --only):
#   1_eda  2a_baseline  2b_feature_pca  3_5_vqc
#   6_7_uq  8_9_qfl  10_11_external_val
# =============================================================================

# NOTE: pipefail is intentionally NOT set globally — tee pipes would cause
# false failures. Each stage captures Python's exit code explicitly instead.
set -uo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
LOG_DIR="./pipeline_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# ── Stage definitions (order matters) ────────────────────────────────────────
declare -a STAGES=(
    "1_eda"
    "2a_baseline"
    "2b_feature_pca"
    "3_5_vqc"
    "6_7_uq"
    "8_9_qfl"
    "10_11_external_val"
)

declare -A STAGE_SCRIPTS=(
    ["1_eda"]="1_eda.py"
    ["2a_baseline"]="2a_baseline.py"
    ["2b_feature_pca"]="2b_feature_pca.py"
    ["3_5_vqc"]="3_5_vqc.py"
    ["6_7_uq"]="6_7_uq.py"
    ["8_9_qfl"]="8_9_qfl.py"
    ["10_11_external_val"]="10_11_external_val.py"
)

declare -A STAGE_DESC=(
    ["1_eda"]="1:     Data audit & EDA (Mendeley + KAU-BCMD)"
    ["2a_baseline"]="2a:    Classical baseline (MobileNetV2 progressive fine-tuning)"
    ["2b_feature_pca"]="2b:    Feature extraction + PCA (bridge to quantum)"
    ["3_5_vqc"]="3-5:  VQC design, Regime A/B, hyperparameter sweep, noise robustness"
    ["6_7_uq"]="6-7:  Uncertainty quantification (MC-Dropout + quantum shot variance)"
    ["8_9_qfl"]="8-9:  Simulated Quantum Federated Learning (QFL) + DP trade-off"
    ["10_11_external_val"]="10-11: External validation (KAU-BCMD) + master ablation table"
)

# ── Parse arguments ───────────────────────────────────────────────────────────
FROM_STAGE=""
ONLY_STAGE=""
SKIP_SLOW=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)      FROM_STAGE="$2"; shift 2 ;;
        --only)      ONLY_STAGE="$2"; shift 2 ;;
        --skip-slow) SKIP_SLOW=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--from STAGE] [--only STAGE] [--skip-slow]"
            echo "Valid stage names: ${STAGES[*]}"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate --from / --only stage names early ───────────────────────────────
validate_stage_name() {
    local name="$1" flag="$2"
    for s in "${STAGES[@]}"; do
        [[ "$s" == "$name" ]] && return 0
    done
    echo "ERROR: '$name' is not a valid stage name for $flag"
    echo "Valid stages: ${STAGES[*]}"
    exit 1
}
[[ -n "$FROM_STAGE" ]]  && validate_stage_name "$FROM_STAGE" "--from"
[[ -n "$ONLY_STAGE" ]]  && validate_stage_name "$ONLY_STAGE" "--only"

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

# Colour codes
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

log()       { echo -e "$1" | tee -a "$MASTER_LOG"; }
log_stage() {
    log ""
    log "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    log "${BOLD}${BLUE}  $1${NC}"
    log "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}
log_ok()   { log "${GREEN}  ✓ $1${NC}"; }
log_err()  { log "${RED}  ✗ $1${NC}"; }
log_warn() { log "${YELLOW}  ⚠ $1${NC}"; }
log_info() { log "  $1"; }

# ── Check dependencies ────────────────────────────────────────────────────────
check_deps() {
    log_stage "Checking dependencies"
    local missing=()

    $PYTHON -c "import torch"       2>/dev/null || missing+=("torch")
    $PYTHON -c "import pennylane"   2>/dev/null || missing+=("pennylane")
    $PYTHON -c "import flwr"        2>/dev/null || missing+=("flwr")
    $PYTHON -c "import sklearn"     2>/dev/null || missing+=("scikit-learn")
    $PYTHON -c "import torchvision" 2>/dev/null || missing+=("torchvision")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_warn "Missing packages: ${missing[*]}"
        log_info "Installing from requirements.txt..."
        $PYTHON -m pip install -r requirements.txt -q
        log_ok "Dependencies installed"
    else
        log_ok "All dependencies present"
    fi

    if [[ ! -f "cache_check.py" ]]; then
        log_err "cache_check.py not found in current directory"
        log_info "Ensure all pipeline scripts are in the same directory as run_pipeline.sh"
        exit 1
    fi

    # Verify every stage script exists before starting any work
    local missing_scripts=()
    for stage in "${STAGES[@]}"; do
        local script="${STAGE_SCRIPTS[$stage]}"
        [[ ! -f "$script" ]] && missing_scripts+=("$script")
    done
    if [[ ${#missing_scripts[@]} -gt 0 ]]; then
        log_err "Missing script files: ${missing_scripts[*]}"
        log_info "Ensure all .py files are in the same directory as run_pipeline.sh"
        exit 1
    fi
    log_ok "All stage scripts present"

    log_info "Python:    $($PYTHON --version)"
    log_info "PyTorch:   $($PYTHON -c 'import torch; print(torch.__version__)')"
    log_info "PennyLane: $($PYTHON -c 'import pennylane; print(pennylane.__version__)')"
    log_info "Flower:    $($PYTHON -c 'import flwr; print(flwr.__version__)')"
    log_info "GPU:       $($PYTHON -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")' 2>/dev/null || echo 'unknown')"
}

# ── Run a single stage ────────────────────────────────────────────────────────
run_stage() {
    local stage="$1"
    local script="${STAGE_SCRIPTS[$stage]}"
    local desc="${STAGE_DESC[$stage]}"
    local stage_log="${LOG_DIR}/${stage}_${TIMESTAMP}.log"

    log_stage "$desc"
    log_info "Script: $script"
    log_info "Log:    $stage_log"

    local start_time
    start_time=$(date +%s)

    # KEY FIX: capture Python exit code independently of tee
    # Using a temp file for exit code avoids the pipefail/tee problem.
    # The pipe itself is fine; we just don't rely on its exit code.
    local exit_code_file
    exit_code_file=$(mktemp)

    (
        $PYTHON "$script" 2>&1
        echo $? > "$exit_code_file"
    ) | tee "$stage_log" | tee -a "$MASTER_LOG"

    local py_exit
    py_exit=$(cat "$exit_code_file")
    rm -f "$exit_code_file"

    local end_time elapsed
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))

    if [[ "$py_exit" == "0" ]]; then
        log_ok "Stage '${stage}' completed in ${mins}m ${secs}s"
        echo "${stage}:SUCCESS:${mins}m${secs}s" >> "${LOG_DIR}/stage_times_${TIMESTAMP}.txt"
        return 0
    else
        log_err "Stage '${stage}' FAILED (exit code: ${py_exit}) after ${mins}m ${secs}s"
        log_info "Full log: $stage_log"
        log_info "To resume from this stage: ./run_pipeline.sh --from ${stage}"
        echo "${stage}:FAILED:exit${py_exit}" >> "${LOG_DIR}/stage_times_${TIMESTAMP}.txt"
        return 1
    fi
}

# ── Determine which stages to run ─────────────────────────────────────────────
should_run() {
    local stage="$1"

    # --only: run exactly one stage
    if [[ -n "$ONLY_STAGE" ]]; then
        [[ "$stage" == "$ONLY_STAGE" ]] && return 0 || return 1
    fi

    # --from: skip all stages before FROM_STAGE
    if [[ -n "$FROM_STAGE" ]]; then
        local found=false
        for s in "${STAGES[@]}"; do
            [[ "$s" == "$FROM_STAGE" ]] && found=true
            if [[ "$s" == "$stage" && "$found" == true ]]; then
                return 0
            fi
        done
        return 1
    fi

    # Default: run all stages
    return 0
}

# ── Apply skip-slow patch (uses temp copy, never modifies original) ───────────
apply_skip_slow() {
    if [[ "$SKIP_SLOW" == false ]]; then
        return  # nothing to do — full sweep runs by default
    fi

    log_warn "--skip-slow: VQC sweep reduced to single config (original file unchanged)"
    local tmp_script="/tmp/3_5_vqc_slim_${TIMESTAMP}.py"
    $PYTHON - <<EOF
import pathlib
p   = pathlib.Path("3_5_vqc.py")
txt = p.read_text()
txt = txt.replace('"n_qubits":    [4, 6],',        '"n_qubits":    [4],')
txt = txt.replace('"n_layers":    [1, 2, 3],',     '"n_layers":    [2],')
txt = txt.replace('"lr":          [0.01, 0.005],', '"lr":          [0.01],')
pathlib.Path("${tmp_script}").write_text(txt)
print("  Slim config written to: ${tmp_script}")
EOF
    STAGE_SCRIPTS["3_5_vqc"]="$tmp_script"
    log_info "3_5_vqc will run from: $tmp_script"
}

# ── Main pipeline ─────────────────────────────────────────────────────────────
main() {
    log "═════════════════════════════════════════════════════════════════════════"
    log "${BOLD}  QML Pipeline Orchestrator${NC}"
    log "  Privacy-Preserving Quantum Federated Learning"
    log "  Breast Cancer Screening — African & MENA Populations"
    log "  Started: $(date)"
    log "  Log:     $MASTER_LOG"
    log "  SKIP_SLOW: $SKIP_SLOW | FROM: '${FROM_STAGE}' | ONLY: '${ONLY_STAGE}'"
    log "═════════════════════════════════════════════════════════════════════════"

    check_deps
    apply_skip_slow

    local failed_stages=()
    local skipped_stages=()
    local completed_stages=()
    local pipeline_start
    pipeline_start=$(date +%s)

    for stage in "${STAGES[@]}"; do
        if should_run "$stage"; then
            if run_stage "$stage"; then
                completed_stages+=("$stage")
            else
                failed_stages+=("$stage")
                log_err "Pipeline halted at stage: $stage"
                log_info "Resume: mkdir -p pipeline_logs && nohup bash run_pipeline.sh --from $stage > pipeline_logs/nohup_resume.log 2>&1 &"
                break
            fi
        else
            skipped_stages+=("$stage")
            log_info "Skipped: $stage"
        fi
    done

    # ── Final summary ──────────────────────────────────────────────────────
    local pipeline_end elapsed_total
    pipeline_end=$(date +%s)
    elapsed_total=$((pipeline_end - pipeline_start))
    local hours=$((elapsed_total / 3600))
    local mins=$(( (elapsed_total % 3600) / 60 ))
    local secs=$((elapsed_total % 60))

    log ""
    log "═════════════════════════════════════════════════════════════════════════"
    log "${BOLD}  PIPELINE SUMMARY${NC}"
    log "  Total time: ${hours}h ${mins}m ${secs}s"
    log "  Completed:  ${#completed_stages[@]} / ${#STAGES[@]} stages"
    [[ ${#completed_stages[@]} -gt 0 ]] && log_ok  "Completed: ${completed_stages[*]}"
    [[ ${#skipped_stages[@]}   -gt 0 ]] && log_warn "Skipped:   ${skipped_stages[*]}"
    [[ ${#failed_stages[@]}    -gt 0 ]] && log_err  "Failed:    ${failed_stages[*]}"

    log ""
    log "  Stage timing: ${LOG_DIR}/stage_times_${TIMESTAMP}.txt"
    log "  Master log:   $MASTER_LOG"
    log ""
    log "  Key output directories:"
    log "    EDA:            ./eda_outputs/"
    log "    Classical:      ./baseline_outputs/"
    log "    Features/PCA:   ./feature_outputs/"
    log "    VQC (3-5):      ./vqc_outputs/"
    log "    UQ  (6-7):      ./uq_outputs/"
    log "    QFL (8-9):      ./qfl_outputs/"
    log "    External val:   ./external_val_outputs/"
    log "═════════════════════════════════════════════════════════════════════════"

    [[ ${#failed_stages[@]} -gt 0 ]] && exit 1
    log_ok "All pipeline stages completed successfully"
}

main "$@"