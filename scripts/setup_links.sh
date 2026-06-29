#!/usr/bin/env bash
# --- link_datasets.sh ---


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_PARENT_DIR="$(dirname "$PROJECT_DIR")"
DATASETS_DIR="${PROJECT_DIR}/datasets"


if [ ! -d "$DATASETS_DIR" ]; then
    echo "Error: Directory $DATASETS_DIR does not exist."
    exit 1
fi

cd "$DATASETS_DIR"
ln -s TID2013 tid2013
ln -s KoNViD-1k konvid-1k
ln -s T2VQA-DB t2vqa-db

ls -la | grep "^l"