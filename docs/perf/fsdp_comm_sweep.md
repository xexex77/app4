# FSDP comm sweep — `ttt_llama_47b` @ seq_len=2048 (8×B200)

Goal: reduce **NCCL comm overhead** (ReduceScatter/AllGather dominates) and achieve **≥15% end-to-end toks/s** at `seq_len=2048`, or document why we’re capped.

## Context (from profiling)

From `docs/perf/bottlenecks.md`, GPU kernel time was dominated by NCCL collectives, e.g.:
- `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(...)` ~45.9%
- `ncclDevKernel_AllGather_RING_LL(...)` ~11.2%

This suggests LN micro-optimizations won’t move the needle much; we need to change **FSDP comm behavior** (collective count, overlap, and/or resharding strategy).

## What changed in code

We added FSDP comm knobs to `app4/ttt/train/train.py` and `app4/ttt/train/fsdp.py`:
- `--fsdp-wrap {auto,block,block2}`
  - `auto`: transformer auto-wrap
  - `block`: wrap each `TTTLlamaBlock`
  - `block2`: **single-module wrapping** (no inner auto-wrap; fewer collectives, potentially higher peak mem)
- `--fsdp-bucket-cap-mb`: best-effort (ignored if unsupported by this torch build)
- `--fsdp-reshard-after-forward {0,1}`:
  - `1`: `FULL_SHARD` (reshard after fwd)
  - `0`: `SHARD_GRAD_OP` if available (keeps params resident; fewer all-gathers; more memory)
- `--fsdp-limit-all-gathers {0,1}`: best-effort (ignored if unsupported)

## Benchmark harness

We added a controlled benchmark module:

```bash
python -m app4.ttt.utils.bench_trainstep \
  --config ttt_llama_47b --seq-len 2048 --steps 30 --warmup 5 \
  --strategy fsdp --precision bf16 \
  --dataset /home/ubuntu/datasets/tinyshakespeare.txt --tokenizer bytes \
  --fsdp-wrap auto \
  --fsdp-reshard-after-forward 1 \
  --fsdp-limit-all-gathers 1 \
  --fsdp-bucket-cap-mb 0
```

Metric window:
- Average `toks/s` and `step_s` over **steps 5–24** (20 steps).
- Peak memory from `torch.cuda.max_memory_allocated()` (rank0).

## Sweep plan (small + disciplined)

Matrix:
- wrap: `auto` vs `block`
- bucket_cap_mb: `25, 50, 100, 200` (if supported; otherwise logged as ignored)
- reshard_after_forward: `1`, then `0` **only if memory allows**

Stop early if peak memory becomes unsafe.

## Results

This sweep is intended to be run on the 8×B200 node when GPUs are available (single-node only).

| wrap | bucket_cap_mb | reshard_after_forward | limit_all_gathers | toks/s avg | step_s avg | peak_mem_gb | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| (TODO) | (TODO) | (TODO) | (TODO) | (TODO) | (TODO) | (TODO) | (TODO) |

## If we still can’t reach +15%

Record:
- best-achieved toks/s vs baseline
- profiler % of NCCL kernels under the best setting (trace + summary)
- whether the bottleneck is network/collective latency vs compute overlap

