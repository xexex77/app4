# Minimal eval: Tiny Shakespeare (bytes) — seq_len=2048

This document records a minimal held-out evaluation on Tiny Shakespeare for the 47B bring-up runs.

## Setup

- Git SHA: `7e3ef21808a09deb928f1aa1cb2968e58f768f60`
- Config: `ttt_llama_47b`
- Strategy: `--strategy fsdp`
- Precision: `--precision bf16`
- Tokenizer: `--tokenizer bytes`
- Dataset: `/home/ubuntu/datasets/tinyshakespeare.txt`
- Sequence length: `--seq-len 2048`
- Batch size: `--batch-size 1`
- Eval split: `--eval-split 0.1` (last 10% of packed token stream)
- Eval batches: `--eval-steps 5` **per rank**  
  - With `world_size=8`, the script all-reduces, so logs show `eval_steps=40`.

## Commands

Step 150 checkpoint:

```bash
CKPT_BASE=~/checkpoints/step3_ts2048_tinyshakespeare
RESUME=$CKPT_BASE/step_000000150

torchrun --standalone --nproc_per_node=8 -m app4.ttt.train.train \
  --config ttt_llama_47b \
  --strategy fsdp --precision bf16 \
  --seq-len 2048 --batch-size 1 \
  --dataset ~/datasets/tinyshakespeare.txt --tokenizer bytes \
  --ckpt-dir "$CKPT_BASE" --resume "$RESUME" \
  --eval-only --eval-steps 5 --eval-split 0.1
```

Step 200 checkpoint:

```bash
CKPT_BASE=~/checkpoints/step3_ts2048_tinyshakespeare
RESUME=$CKPT_BASE/step_000000200

torchrun --standalone --nproc_per_node=8 -m app4.ttt.train.train \
  --config ttt_llama_47b \
  --strategy fsdp --precision bf16 \
  --seq-len 2048 --batch-size 1 \
  --dataset ~/datasets/tinyshakespeare.txt --tokenizer bytes \
  --ckpt-dir "$CKPT_BASE" --resume "$RESUME" \
  --eval-only --eval-steps 5 --eval-split 0.1
```

Raw logs (not committed):
- `/home/ubuntu/runs/logs/eval_ts2048_tinyshakespeare_step150.log`
- `/home/ubuntu/runs/logs/eval_ts2048_tinyshakespeare_step200.log`

## Results

| checkpoint | eval loss | perplexity |
| --- | ---: | ---: |
| step_150 | 116.2693 | ~3.1268e50 |
| step_200 | 34.4840 | ~9.4665e14 |

Trend: **eval loss decreased from step 150 → step 200**, as expected for a short continued-train window (finite metrics, improving direction).

