#!/usr/bin/env bash
#
# Deploy code to a RunPod GPU instance, run experiments, and pull results.
#
# Usage:
#   bash scripts/deploy.sh push       # copy code to pod
#   bash scripts/deploy.sh run        # install deps + run full pipeline
#   bash scripts/deploy.sh pull       # copy results back to local
#   bash scripts/deploy.sh full       # push + run + pull (end-to-end)
#   bash scripts/deploy.sh debug      # push + debug-fast + pull
#   bash scripts/deploy.sh clean      # push + clean checkpoints/cache on pod + run
#   bash scripts/deploy.sh resume     # push + resume from where last run left off
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── SSH Configuration ────────────────────────────────────────────────────────
SSH_HOST="root@80.15.7.37"
SSH_PORT="39047"
SSH_KEY="$HOME/.ssh/id_ed25519"
# Use RunPod's persistent volume. /root is container storage and can be lost or
# too small; /workspace is where enlarged RunPod volumes are mounted.
REMOTE_DIR="/workspace/NLP-Project1"

SSH_CMD="ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 $SSH_HOST"
RSYNC_SSH="ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no"

info()  { echo "  [$(date +%H:%M:%S)] $*"; }
ok()    { echo "  [OK] $*"; }
warn()  { echo "  [WARN] $*"; }
fail()  { echo "  [FAIL] $*"; }

# ── Push ──────────────────────────────────────────────────────────────────────
push_code() {
    echo ""
    echo "============================================================"
    echo "  PUSH: Syncing code to $SSH_HOST"
    echo "============================================================"
    echo ""

    info "Testing SSH connection..."
    if $SSH_CMD "echo 'SSH connection OK'" 2>/dev/null; then
        ok "SSH connection works"
    else
        fail "Cannot connect to $SSH_HOST on port $SSH_PORT"
        exit 1
    fi

    info "Ensuring remote directory exists..."
    $SSH_CMD "mkdir -p $REMOTE_DIR"

    info "Syncing project files..."
    rsync -avz --progress \
        -e "$RSYNC_SSH" \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='.cache/' \
        --exclude='.venv/' \
        --exclude='results/checkpoints/' \
        --exclude='results/figures/' \
        --exclude='results/logs/' \
        --exclude='results/benchmarks/' \
        --exclude='figures/' \
        --exclude='*.pyc' \
        --exclude='.DS_Store' \
        --exclude='results/metrics/*.json' \
        "${PROJECT_DIR}/" "${SSH_HOST}:${REMOTE_DIR}/"

    ok "Code pushed to $SSH_HOST:$REMOTE_DIR"
    echo ""
}

# ── Remote setup ──────────────────────────────────────────────────────────────
remote_setup() {
    echo ""
    echo "============================================================"
    echo "  SETUP: Verifying environment on remote pod"
    echo "============================================================"
    echo ""

    # Verify PyTorch + CUDA (the image provides them)
    info "Verifying PyTorch + CUDA..."
    $SSH_CMD "python -c '
import torch
print(f\"  PyTorch:   {torch.__version__}\")
print(f\"  CUDA:      {torch.cuda.is_available()}\")
if torch.cuda.is_available():
    print(f\"  GPU:       {torch.cuda.get_device_name(0)}\")
    print(f\"  VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\")
else:
    print(\"  *** FATAL: CUDA NOT AVAILABLE! ***\")
'" 2>&1 | head -10

    # Show disk space
    info "Available disk space on pod:"
    $SSH_CMD "df -h /workspace 2>/dev/null | tail -1 || df -h / | tail -1" 2>&1

    # Install missing project deps (skip torch — image provides it)
    info "Installing project dependencies..."
    $SSH_CMD "pip install --break-system-packages --quiet transformers datasets evaluate scikit-learn pandas numpy matplotlib seaborn tqdm accelerate pyyaml safetensors 2>&1 | tail -3"

    ok "Environment setup complete"
    echo ""
}

# ── Run ────────────────────────────────────────────────────────────────────────
run_experiments() {
    remote_setup

    echo ""
    echo "============================================================"
    echo "  RUN: Starting experiment pipeline on GPU pod"
    echo "============================================================"
    echo ""

    info "Starting pipeline..."
    echo ""
    echo "  Monitor with:"
    echo "    $SSH_CMD 'cd $REMOTE_DIR && tail -f results/logs/run_*/<log_name>.log'"
    echo ""

    $SSH_CMD "cd $REMOTE_DIR && BATCH_SIZE=128 bash scripts/run_all.sh" 2>&1 | tee /tmp/runpod_pipeline.log

    ok "Pipeline finished (check logs for any errors)"
    echo ""
}

run_debug() {
    remote_setup

    echo ""
    echo "============================================================"
    echo "  DEBUG: Running debug-fast pipeline on GPU pod (~10 min)"
    echo "============================================================"
    echo ""

    $SSH_CMD "cd $REMOTE_DIR && BATCH_SIZE=64 bash scripts/run_all.sh --debug-fast" 2>&1 | tee /tmp/runpod_debug.log

    ok "Debug pipeline finished"
    echo ""
}

run_clean_resume() {
    remote_setup

    echo ""
    echo "============================================================"
    echo "  CLEAN + RESUME: Clean disk then resume pipeline"
    echo "============================================================"
    echo ""

    info "Cleaning checkpoints and cache on pod..."
    $SSH_CMD "cd $REMOTE_DIR && find results/checkpoints -maxdepth 3 -type d -name 'checkpoint-*' -exec rm -rf {} + 2>/dev/null; find results/checkpoints -type f \( -name 'optimizer_state.pt' -o -name 'rng_state_*.pth' -o -name 'scheduler.pt' -o -name 'training_args.bin' -o -name 'trainer_state.json' \) -delete 2>/dev/null; echo 'Cleanup done'"
    $SSH_CMD "df -h /workspace 2>/dev/null | tail -1 || df -h / | tail -1" 2>&1

    info "Starting pipeline with resume..."
    $SSH_CMD "cd $REMOTE_DIR && BATCH_SIZE=128 bash scripts/run_all.sh" 2>&1 | tee /tmp/runpod_pipeline.log

    ok "Pipeline finished"
    echo ""
}

# ── Pull ───────────────────────────────────────────────────────────────────────
pull_results() {
    echo ""
    echo "============================================================"
    echo "  PULL: Downloading results from GPU pod"
    echo "============================================================"
    echo ""

    mkdir -p "${PROJECT_DIR}/results/metrics" \
             "${PROJECT_DIR}/results/tables" \
             "${PROJECT_DIR}/results/figures" \
             "${PROJECT_DIR}/results/logs" \
             "${PROJECT_DIR}/results/benchmarks" \
             "${PROJECT_DIR}/figures"

    LOCAL_DIRS=(
        "results/metrics"
        "results/tables"
        "results/figures"
        "results/logs"
        "results/benchmarks"
        "figures"
    )

    pulled=0
    for DIR in "${LOCAL_DIRS[@]}"; do
        info "Pulling $DIR..."
        if rsync -avz --progress \
            -e "$RSYNC_SSH" \
            "${SSH_HOST}:${REMOTE_DIR}/${DIR}/" \
            "${PROJECT_DIR}/${DIR}/" 2>/dev/null; then
            pulled=$((pulled+1))
        else
            warn "No files in $DIR or rsync error"
        fi
    done

    echo ""
    ok "Results pulled ($pulled directories synced)"
    echo ""
    echo "  Regenerate local output:"
    echo "    python -m src.make_results_summary"
    echo "    python -m src.build_report_assets"
    echo "    python -m src.visualize --results-dir results --figures-dir figures"
    echo ""
}

# ── Main ───────────────────────────────────────────────────────────────────────
case "${1:-}" in
    push)
        push_code
        ;;
    run)
        run_experiments
        ;;
    debug)
        push_code
        run_debug
        pull_results
        ;;
    pull)
        pull_results
        ;;
    full)
        push_code
        run_experiments
        pull_results
        ;;
    clean)
        push_code
        run_clean_resume
        pull_results
        ;;
    resume)
        push_code
        run_clean_resume
        pull_results
        ;;
    *)
        echo ""
        echo "Usage: bash scripts/deploy.sh {push|run|pull|full|debug|clean|resume}"
        echo ""
        echo "  push   - Sync code to RunPod pod"
        echo "  run    - Verify env + run full pipeline"
        echo "  pull   - Download results from pod"
        echo "  full   - push + run + pull"
        echo "  debug  - push + debug-fast + pull"
        echo "  clean  - push + clean disk + resume pipeline"
        echo "  resume - push + clean disk + resume + pull"
        echo ""
        echo "  Pod: RTX 5090 (32GB), runpod/pytorch:1.0.2-cu1281-torch280"
        echo ""
        exit 1
        ;;
esac
