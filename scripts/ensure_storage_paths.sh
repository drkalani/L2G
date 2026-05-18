#!/usr/bin/env bash

set -euo pipefail

HOME_DIR="${HOME:-/root}"
L2G_DATA_DIR="${L2G_DATA_DIR:-$HOME_DIR/l2g-data}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME_DIR/.cache/huggingface}"

mkdir -p "$L2G_DATA_DIR" "$HF_CACHE_DIR"

if [ ! -d "$L2G_DATA_DIR" ] || [ ! -d "$HF_CACHE_DIR" ]; then
  echo "[ERROR] Failed to create required storage directories."
  exit 1
fi

touch "$L2G_DATA_DIR/.writetest" 2>/dev/null
rm -f "$L2G_DATA_DIR/.writetest"
touch "$HF_CACHE_DIR/.writetest" 2>/dev/null
rm -f "$HF_CACHE_DIR/.writetest"

echo "[OK] Storage paths are ready:"
echo "  L2G_DATA_DIR=$L2G_DATA_DIR"
echo "  HF_CACHE_DIR=$HF_CACHE_DIR"
