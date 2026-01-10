#!/usr/bin/env bash
set -euo pipefail

# DeepSpeed integration is wired but minimal in this skeleton; tune DS config for production.
deepspeed --num_gpus=8 -m app4.ttt.train.train \
  --config ttt_llama_47b \
  --strategy deepspeed \
  --precision bf16 \
  --synthetic \
  --steps 200 \
  --seq-len 1024 \
  --batch-size 1

