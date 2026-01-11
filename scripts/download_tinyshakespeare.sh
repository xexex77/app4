#!/usr/bin/env bash
set -euo pipefail

URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Prefer /home/ubuntu/work/datasets when running on the GPU box, but fall back to ./datasets.
DEFAULT_DIR="/home/ubuntu/work/datasets"
OUT_DIR="${1:-$DEFAULT_DIR}"

if [[ "$OUT_DIR" == "$DEFAULT_DIR" ]]; then
  if ! mkdir -p "$OUT_DIR" 2>/dev/null; then
    OUT_DIR="datasets"
    mkdir -p "$OUT_DIR"
  fi
else
  mkdir -p "$OUT_DIR"
fi

OUT_FILE="$OUT_DIR/tinyshakespeare.txt"
if [[ -f "$OUT_FILE" ]]; then
  echo "Already exists: $OUT_FILE"
  exit 0
fi

echo "Downloading Tiny Shakespeare -> $OUT_FILE"
curl -L --fail -o "$OUT_FILE" "$URL"
echo "Done: $OUT_FILE ($(wc -c <\"$OUT_FILE\") bytes)"

