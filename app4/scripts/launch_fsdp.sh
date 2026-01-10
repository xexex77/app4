#!/usr/bin/env bash
set -euo pipefail

torchrun --standalone --nproc_per_node=8 -m app4.ttt.train.train \
  --config ttt_llama_47b \
  --strategy fsdp2 \
  --precision bf16 \
  --synthetic \
  --steps 200 \
  --seq-len 1024 \
  --batch-size 1

