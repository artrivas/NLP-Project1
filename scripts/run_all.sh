#!/usr/bin/env bash
#
# Resume experiment pipeline — skips already-completed runs.
#
# This script is idempotent: it checks for existing results and checkpoints
# before re-running, so it's safe to run after a partial failure.
#
# Disk management:
#   - Deletes intermediate Trainer checkpoints after each run (keeps best_model/)
#   - Cleans HuggingFace model cache after downloads complete
#   - Frees /tmp periodically
#
# Usage:
#   bash scripts/run_all.sh                # full run / resume
#   bash scripts/run_all.sh --debug-fast   # smoke test
#   bash scripts/run_all.sh --clean        # clean up checkpoints and cache first
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

MODE="--full-run"
CLEAN_FIRST=false
if [[ "${1:-}" == "--debug-fast" ]]; then
    MODE="--debug-fast"
elif [[ "${1:-}" == "--clean" ]]; then
    CLEAN_FIRST=true
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHON_PATH:-}"
if [[ -d "/workspace" ]]; then
    export HF_HOME="/workspace/.cache/huggingface"
    export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"
    export TMPDIR="/workspace/tmp"
else
    export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
    export HF_DATASETS_CACHE="${PROJECT_DIR}/.cache/huggingface/datasets"
    export TMPDIR="${PROJECT_DIR}/tmp"
fi
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

BATCH_SIZE="${BATCH_SIZE:-128}"
MIN_FREE_GB="${MIN_FREE_GB:-12}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/logs/run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

METRICS_DIR="results/metrics"
CKPT_DIR="results/checkpoints"

TOTAL_STEPS=9
step=0

log()      { echo ""; echo "============================================================"; echo "[$(date +%H:%M:%S)] $*"; echo "============================================================"; }
progress() { step=$((step+1)); echo ""; echo ">>>>>>>>>> STEP $step/$TOTAL_STEPS: $* <<<<<<<<<<"; }
info()     { echo "  [$(date +%H:%M:%S)] $*"; }
ok()       { echo "  [OK] $*"; }
fail()     { echo "  [FAIL] $*"; }
skip()     { echo "  [SKIP] $* (already complete)"; }

# ── Cleanup functions ──────────────────────────────────────────────────────────

clean_checkpoints() {
    # Remove intermediate Trainer checkpoints, keep only best_model/
    info "Cleaning intermediate trainer checkpoints..."
    find "$CKPT_DIR" -maxdepth 5 -type d -name 'checkpoint-*' -exec rm -rf {} + 2>/dev/null || true
    # Also clean optimizer/rng states in best_model dirs (we only need model weights)
    find "$CKPT_DIR" -type f \( -name 'optimizer_state.pt' -o -name 'rng_state_*.pth' -o -name 'scheduler.pt' -o -name 'trainer_state.json' -o -name 'training_args.bin' \) -delete 2>/dev/null || true
    info "Checkpoint cleanup done."
}

clean_hf_cache() {
    # Keep only model weights in HF cache, remove downloaded model files we don't need
    info "Cleaning HuggingFace cache..."
    find "$HF_HOME" -type f -name '*.lock' -delete 2>/dev/null || true
    # Remove model repos we've already loaded (they'll re-download if needed, but checkpoints exist)
    if [[ "$MODE" == "--full-run" ]]; then
        # After all baselines + ablations are done, the checkpoints exist so we can clear
        # But during training we need the tokenizer files — so only clean large items
        find "$HF_HOME" -type f -name '*.msgpack' -delete 2>/dev/null || true
    fi
    info "HF cache cleanup done."
}

free_disk() {
    # Show available disk space
    local avail
    avail=$(df -h . | awk 'NR==2{print $4}')
    info "Available disk space: $avail"
}

check_disk_or_die() {
    local free_gb
    free_gb=$(df -BG . | awk 'NR==2{gsub("G", "", $4); print $4}')
    if [[ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]]; then
        echo ""
        echo "FATAL: only ${free_gb}GB free; refusing to continue."
        echo "Need at least ${MIN_FREE_GB}GB free for checkpoints + temp files."
        echo "Run: bash scripts/deploy.sh clean"
        exit 1
    fi
}

# ── Check completion functions ─────────────────────────────────────────────────

# Check if a metric row exists in the CSV for a given dataset+run
has_metric() {
    local csv="$1"
    local dataset="$2"
    local column="$3"
    local value="$4"
    if [[ ! -f "$csv" ]]; then
        return 1
    fi
    grep -q "${dataset},${value}" "$csv" 2>/dev/null || grep -q "${value}" "$csv" 2>/dev/null
}

has_ablation() {
    local dataset="$1"
    local config="$2"
    if [[ ! -f "$METRICS_DIR/distilbert_ablation_metrics.csv" ]]; then
        return 1
    fi
    python -c "
import pandas as pd
try:
    df = pd.read_csv('$METRICS_DIR/distilbert_ablation_metrics.csv')
    ds = df['dataset_run_name'] if 'dataset_run_name' in df.columns else df.get('dataset_name','')
    cfg = df['ablation_config']
    mask = (ds == '${dataset}') & (cfg == '${config}')
    if 'train_batch_size' in df.columns:
        mask = mask & (df['train_batch_size'].astype(str) == '$BATCH_SIZE')
    print('EXISTS' if mask.any() else 'MISSING')
except Exception:
    print('MISSING')
" 2>/dev/null | grep -q EXISTS
}

has_checkpoint() {
    local path="$1"
    [[ -f "$path/best_model/model.safetensors" ]] || [[ -f "$path/model.safetensors" ]]
}

has_baseline_metric() {
    local family="$1"
    local dataset="$2"
    local csv="$METRICS_DIR/${family}_metrics.csv"
    [[ "$family" == "distilbert" ]] && csv="$METRICS_DIR/distilbert_baseline_metrics.csv"
    if [[ ! -f "$csv" ]]; then
        return 1
    fi
    python -c "
import pandas as pd
try:
    df = pd.read_csv('$csv')
    mask = (df.get('dataset_run_name') == '$dataset') & (df.get('model_type') == '$family')
    if 'train_batch_size' in df.columns:
        mask = mask & (df['train_batch_size'].astype(str) == '$BATCH_SIZE')
    ok = mask.any()
    print('EXISTS' if ok else 'MISSING')
except Exception:
    print('MISSING')
" 2>/dev/null | grep -q EXISTS
}

has_complete_baseline() {
    local family="$1"
    local dataset="$2"
    has_baseline_metric "$family" "$dataset" && has_checkpoint "$CKPT_DIR/$family/$dataset"
}

has_complete_ablation() {
    local dataset="$1"
    local config="$2"
    has_ablation "$dataset" "$config" && has_checkpoint "$CKPT_DIR/distilbert_ablation/$dataset/$config"
}

# ── Initial cleanup if requested ───────────────────────────────────────────────

if [[ "$CLEAN_FIRST" == true ]]; then
    progress "Disk cleanup"
    clean_checkpoints
    clean_hf_cache
    free_disk
fi

# ──────────────────────────────────────────────
# Step 0: Environment check
# ──────────────────────────────────────────────
progress "Environment check"
free_disk

python -c "
import torch, sys
print(f'  Python:         {sys.version.split()[0]}')
print(f'  PyTorch:        {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'  GPU count:      {torch.cuda.device_count()}')
else:
    print('  *** FATAL: CUDA is NOT available! ***')
    print('  *** The pipeline requires a CUDA GPU. Aborting. ***')
import transformers
print(f'  Transformers:   {transformers.__version__}')
" 2>&1 | head -20
info "Pipeline mode: $MODE"
info "Batch size override: $BATCH_SIZE"
info "HF_HOME: $HF_HOME"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FATAL: CUDA not available. Aborting."
    exit 1
fi
ok "Environment ready"

# ──────────────────────────────────────────────
# Step 1: DistilBERT baselines (3 datasets)
# ──────────────────────────────────────────────
progress "DistilBERT baselines (3 datasets)"
DATASETS=(ag_news sst2 yelp_review_full)

for DS in "${DATASETS[@]}"; do
    check_disk_or_die
    if has_complete_baseline "distilbert" "$DS"; then
        skip "DistilBERT baseline on $DS"
        continue
    fi
    info "Training DistilBERT baseline on $DS..."
    if python -m src.train \
        --config "config/${DS}.yaml" \
        --model distilbert-base-uncased \
        $MODE --fp16 --batch-size "$BATCH_SIZE" \
        2>&1 | tee "$LOG_DIR/distilbert_baseline_${DS}.log"; then
        ok "DistilBERT baseline on $DS complete"
    else
        fail "DistilBERT baseline on $DS failed"
    fi
    # Free disk: remove intermediate checkpoints, keep best_model
    clean_checkpoints
    free_disk
done

# ──────────────────────────────────────────────
# Step 2: DistilBERT ablations (5 configs × 3 datasets)
# ──────────────────────────────────────────────
progress "DistilBERT ablations (5 configs × 3 datasets = 15 runs)"
ABLATION_CONFIGS=(baseline frozen_transformer small_classifier large_classifier freeze_lower_layers)
run_num=0
total_runs=15

for DS in "${DATASETS[@]}"; do
    for CFG in "${ABLATION_CONFIGS[@]}"; do
        run_num=$((run_num+1))
        check_disk_or_die
        if has_complete_ablation "$DS" "$CFG"; then
            skip "Ablation $CFG on $DS [$run_num/$total_runs]"
            continue
        fi
        if has_ablation "$DS" "$CFG" && ! has_checkpoint "$CKPT_DIR/distilbert_ablation/$DS/$CFG"; then
            info "Metrics exist for ablation $CFG on $DS, but checkpoint is missing; retraining for benchmarkable checkpoint."
        fi
        info "[${run_num}/${total_runs}] Ablation $CFG on $DS..."
        if python -m src.ablation \
            --dataset-config "config/${DS}.yaml" \
            --ablation-config config/ablation_distilbert.yaml \
            --config-name "$CFG" \
            $MODE --fp16 --batch-size "$BATCH_SIZE" \
            2>&1 | tee "$LOG_DIR/ablation_${DS}_${CFG}.log"; then
            ok "Ablation $CFG on $DS complete"
        else
            fail "Ablation $CFG on $DS failed"
        fi
        clean_checkpoints
        free_disk
    done
done

# ──────────────────────────────────────────────
# Step 3: Select best ablation config per dataset
# ──────────────────────────────────────────────
progress "Select best ablation config per dataset"
if python -m src.select_best_ablation 2>&1 | tee "$LOG_DIR/select_best.log"; then
    ok "Best ablation selection complete"
    echo "--- Best configs ---"
    cat "$METRICS_DIR/best_distilbert_ablation.csv" 2>/dev/null || true
else
    fail "Best ablation selection failed"
fi

# ──────────────────────────────────────────────
# Step 4: BERT baselines (3 datasets)
# ──────────────────────────────────────────────
progress "BERT baselines (3 datasets)"

for DS in "${DATASETS[@]}"; do
    check_disk_or_die
    if has_complete_baseline "bert" "$DS"; then
        skip "BERT baseline on $DS (checkpoint exists)"
        continue
    fi
    info "Training BERT baseline on $DS..."
    if python -m src.train \
        --config "config/${DS}.yaml" \
        --model bert-base-uncased \
        $MODE --fp16 --batch-size "$BATCH_SIZE" \
        2>&1 | tee "$LOG_DIR/bert_baseline_${DS}.log"; then
        ok "BERT baseline on $DS complete"
    else
        fail "BERT baseline on $DS failed"
    fi
    clean_checkpoints
    free_disk
done

# ──────────────────────────────────────────────
# Step 5: Benchmark all checkpoints
# ──────────────────────────────────────────────
progress "Benchmark all checkpoints"
mkdir -p results/benchmarks

for DS in "${DATASETS[@]}"; do
    CKPT="$CKPT_DIR/distilbert/$DS/best_model"
    if has_checkpoint "$CKPT_DIR/distilbert/$DS"; then
        info "Benchmarking DistilBERT baseline on $DS..."
        python -m src.benchmark \
            --config "config/${DS}.yaml" \
            --model-path "$CKPT" \
            $MODE \
            2>&1 | tee -a "$LOG_DIR/benchmarks.log" || fail "DistilBERT baseline bench on $DS"
    fi

    CKPT="$CKPT_DIR/bert/$DS/best_model"
    if has_checkpoint "$CKPT_DIR/bert/$DS"; then
        info "Benchmarking BERT baseline on $DS..."
        python -m src.benchmark \
            --config "config/${DS}.yaml" \
            --model-path "$CKPT" \
            --model-name bert-base-uncased \
            $MODE \
            2>&1 | tee -a "$LOG_DIR/benchmarks.log" || fail "BERT baseline bench on $DS"
    fi

    for CFG in "${ABLATION_CONFIGS[@]}"; do
        CKPT="$CKPT_DIR/distilbert_ablation/$DS/$CFG/best_model"
        if has_checkpoint "$CKPT_DIR/distilbert_ablation/$DS/$CFG"; then
            info "Benchmarking ablation $CFG on $DS..."
            python -m src.benchmark \
                --config "config/${DS}.yaml" \
                --model-path "$CKPT" \
                $MODE \
                2>&1 | tee -a "$LOG_DIR/benchmarks.log" || fail "Ablation $CFG bench on $DS"
        fi
    done
done

# ──────────────────────────────────────────────
# Step 6: Final comparison
# ──────────────────────────────────────────────
progress "Final comparison (DistilBERT best vs BERT)"
if python -m src.final_comparison \
    $MODE --output-dir results --fp16 \
    2>&1 | tee "$LOG_DIR/final_comparison.log"; then
    ok "Final comparison complete"
    echo "--- Final performance ---"
    cat "$METRICS_DIR/final_comparison_metrics.csv" 2>/dev/null || true
    echo ""
    echo "--- Final efficiency ---"
    cat "$METRICS_DIR/final_efficiency_metrics.csv" 2>/dev/null || true
else
    fail "Final comparison failed"
fi

# ──────────────────────────────────────────────
# Step 7: Consolidate metrics
# ──────────────────────────────────────────────
progress "Consolidate metrics into report tables"
python -m src.aggregate_results 2>&1 | tee "$LOG_DIR/aggregate.log" || fail "Table consolidation had errors"

# ──────────────────────────────────────────────
# Step 8: Generate report figures
# ──────────────────────────────────────────────
progress "Generate report figures"
python -m src.plot_results 2>&1 | tee "$LOG_DIR/plot_results.log" || fail "Plot generation had errors"
python -m src.build_report_assets 2>&1 | tee "$LOG_DIR/build_report_assets.log" || fail "Report asset build had errors"

# ──────────────────────────────────────────────
# Step 9: Extended visualizations
# ──────────────────────────────────────────────
progress "Extended visualizations"
python -m src.visualize \
    --results-dir results \
    --figures-dir results/figures \
    2>&1 | tee "$LOG_DIR/visualize.log" || fail "Extended visualization had errors"

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE!"
echo "============================================================"
free_disk
echo ""
echo "  Log directory:     $LOG_DIR"
echo "  Key metrics:"
echo "    $METRICS_DIR/final_comparison_metrics.csv"
echo "    $METRICS_DIR/final_efficiency_metrics.csv"
echo "    $METRICS_DIR/distilbert_ablation_metrics.csv"
echo "    $METRICS_DIR/best_distilbert_ablation.csv"
echo ""
echo "  To pull results locally:  bash scripts/deploy.sh pull"
echo "============================================================"
