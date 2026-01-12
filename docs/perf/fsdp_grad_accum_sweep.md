# FSDP grad accumulation sweep (no_sync) — `ttt_llama_47b` @ seq_len=2048

Goal: test whether **gradient accumulation** using **FSDP `no_sync()`** improves end-to-end throughput at `seq_len=2048` (8×B200).

## Setup

- Config: `ttt_llama_47b`
- Strategy: `--strategy fsdp`
- Precision: `--precision bf16`
- Tokenizer: `bytes`
- Dataset: `/home/ubuntu/datasets/tinyshakespeare.txt`
- Batch size: `--batch-size 1`
- Sequence length: `--seq-len 2048`
- Steps: `--steps 25` (report window: optimizer steps **5–24**)
- Log cadence: `--log-every 1`
- Device: CUDA (8 GPUs)

Command (N ∈ {1,2,4}):

```bash
torchrun --standalone --nproc_per_node=8 -m app4.ttt.train.train \
  --config ttt_llama_47b \
  --strategy fsdp --precision bf16 \
  --steps 25 --seq-len 2048 --batch-size 1 \
  --dataset /home/ubuntu/datasets/tinyshakespeare.txt --tokenizer bytes \
  --ckpt-dir /home/ubuntu/checkpoints/perf_accumN${N} --resume none --no-save \
  --log-every 1 \
  --grad-accum-steps ${N}
```

Raw logs (not committed):
- N=1: `/home/ubuntu/runs/logs/perf_accumN1_20260112_125931.log`
- N=2: `/home/ubuntu/runs/logs/perf_accumN2_20260112_131554.log`
- N=4: `/home/ubuntu/runs/logs/perf_accumN4_20260112_134215.log`

## Results (window: optimizer steps 5–24)

| N | toks/s avg | step_s avg | peak_mem_gb (max) | notes |
| ---: | ---: | ---: | ---: | --- |
| 1 | 714.12 | 22.944 | 86.52 | baseline |
| 2 | 706.33 | 46.394 | 182.15 | ~-1.09% toks/s vs N=1; very high peak mem |
| 4 | 703.16 | 93.202 | 182.15 | ~-1.53% toks/s vs N=1; very high peak mem |

Takeaways:
- On this workload/config, **N=2 and N=4 did not improve end-to-end toks/s** vs N=1.
- `peak_mem_gb` increased substantially for N>1, leaving very little headroom.
- Checkpointing: **no extra grad-accum state** is required because checkpoints are only written
  at **optimizer-step boundaries** (i.e., not mid-microstep).

