#!/usr/bin/env bash
# --- archive_results.sh ---
# Description: Industrial-grade archiving tool aligned with results/ five subdomains

set -e

# --- 1. Physical path auto-detection ---
# BASE_DIR=$(cd "$(dirname "$0")"; pwd)
BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_NAME=$(basename "$0")

PROJECT_ROOT=$(cd "$BASE_DIR/.."; pwd)
RESULTS_DIR="${PROJECT_ROOT}/results"


TIMESTAMP=$(date +%m%d_%H%M)
SAVE_DIR="$PROJECT_ROOT/archives"
SAVE_NAME="${SAVE_DIR}/EXP_RESULTS_${TIMESTAMP}.tar.gz"
mkdir -p "$SAVE_DIR"

# Default flags
PACK_LOGS=false
PACK_MODELS=false
PACK_PLOTS=false

usage() {
    echo "Usage: $SCRIPT_NAME [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -l, --logs        Package training and script logs (train_logs, scripts_logs)"
    echo "  -m, --models      Package model checkpoints (.pt from model_outputs)"
    echo "  -p, --plots       Package model visualizations (plots, data_analysis)"
    echo "  -a, --all         Package all of the above results"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $SCRIPT_NAME -l -m           # Quick archive: logs and models"
    echo "  $SCRIPT_NAME --all           # Industrial: archive all five subdomains"
    exit 0
}

if [ $# -eq 0 ]; then
    usage
fi

# Argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--logs)
            PACK_LOGS=true
            shift
            ;;
        -m|--models|--checkpoints|--outputs)
            PACK_MODELS=true
            shift
            ;;
        -p|--plots)
            PACK_PLOTS=true
            shift
            ;;
        -a|--all)
            PACK_LOGS=true
            PACK_MODELS=true
            PACK_PLOTS=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "❌ Unknown option: $1"
            usage
            ;;
    esac
done

cd "$PROJECT_ROOT" || exit 1

PACK_LIST=()
PACK_NAMES=()

# --- Asset 1: Log subdomains (train_logs + scripts_logs) ---
if [ "$PACK_LOGS" = true ]; then
    # Package training business logs
    if [ -d "results/train_logs" ]; then
        PACK_LIST+=("results/train_logs")
        PACK_NAMES+=("train_logs")
    fi
    # Package console script logs
    if [ -d "results/scripts_logs" ]; then
        PACK_LIST+=("results/scripts_logs")
        PACK_NAMES+=("scripts_logs")
    fi
fi

# --- Asset 2: Model outputs subdomain ---
if [ "$PACK_MODELS" = true ]; then
    if [ -d "results/model_outputs" ]; then
        PACK_LIST+=("results/model_outputs")
        PACK_NAMES+=("model_outputs")
    else
        echo "⚠️  Warning: model_outputs directory not found under results/, skipping."
    fi
fi

# --- Asset 3: Visualizations and analysis subdomain ---
if [ "$PACK_PLOTS" = true ]; then
    # Package core visualization charts
    if [ -d "results/plots" ]; then
        PACK_LIST+=("results/plots")
        PACK_NAMES+=("plots")
    fi
    # Package data analysis cache
    if [ -d "results/data_analysis" ]; then
        PACK_LIST+=("results/data_analysis")
        PACK_NAMES+=("data_analysis")
    fi
fi

# Empty asset guard
if [ ${#PACK_LIST[@]} -eq 0 ]; then
    echo "❌ Error: No valid assets found for packaging!"
    echo "Please ensure training has been successfully executed and directories exist."
    exit 1
fi

# Generate archive name
echo "=========================================="
echo "📦 Packaging mode: ${PACK_NAMES[@]}"
echo "📂 Project root: $PROJECT_ROOT"
echo "🚀 Output file: $SAVE_NAME"
echo "=========================================="
echo "Precisely archiving selected subdomains from results/..."

tar --exclude='*__pycache__*' \
    --exclude='*.pyc' \
    --exclude='*.tmp' \
    --exclude='.DS_Store' \
    -czf "$SAVE_NAME" \
    "${PACK_LIST[@]}" \
    --ignore-failed-read

if [ $? -eq 0 ]; then
    echo "------------------------------------------"
    echo "✅ Archive successful -> $SAVE_NAME"
    echo "📊 Package size: $(du -h "$SAVE_NAME" | cut -f1)"
    echo "📁 Assets included: ${PACK_NAMES[@]}"
    echo "------------------------------------------"
else
    echo "❌ Archive failed. Please check disk permissions or quota."
    exit 1
fi