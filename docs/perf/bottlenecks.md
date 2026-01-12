# Perf bottlenecks: `ttt_llama_47b` @ seq_len=2048 (8×B200)

This note summarizes **torch.profiler** results at `seq_len=2048` and one measured optimization.

## Setup

- Config: `ttt_llama_47b`
- Strategy: `--strategy fsdp`
- Precision: `--precision bf16`
- Tokenizer: `bytes`
- Dataset: `/home/ubuntu/datasets/tinyshakespeare.txt`
- Batch size: `--batch-size 1`
- World size: `8`

Profiler command (baseline / optimized were the same aside from code changes):

```bash
torchrun --standalone --nproc_per_node=8 -m app4.ttt.train.train \
  --config ttt_llama_47b \
  --strategy fsdp --precision bf16 \
  --steps 12 --seq-len 2048 --batch-size 1 \
  --dataset ~/datasets/tinyshakespeare.txt --tokenizer bytes \
  --ckpt-dir ~/checkpoints/<...> --resume none --no-save \
  --log-every 6 --grad-clip 0 \
  --profile_steps 10 --profile-dir ~/runs/prof/<...>
```

Raw traces (not committed):
- Baseline trace: `/home/ubuntu/runs/prof/prof_ts2048_base3_20260112_055425/129-213-151-122_253016.1768198322087755842.pt.trace.json`
- Optimized trace: `/home/ubuntu/runs/prof/prof_ts2048_opt3_20260112_061824/129-213-151-122_259356.1768199695042806441.pt.trace.json`

How hotspots were computed:
- Sum `dur` over `traceEvents` where `cat == "kernel"` and `ph == "X"` (GPU kernel events).
- Percentages are relative to total kernel time in that trace.

## Top 3 hotspots (baseline)

1. **`ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(...)` (~45.9%)**
   - Bound: **comm-bound**
   - Candidate: **overlap / reduce FSDP communication** (e.g., FSDP prefetch settings, limiting all-gathers) and verify with a follow-up trace.

2. **`ncclDevKernel_AllGather_RING_LL(...)` (~11.2%)**
   - Bound: **comm-bound**
   - Candidate: **reduce all-gather frequency / overlap** (FSDP settings; ensure comm/compute overlap is effective at 2048).

3. **`at::native::elementwise_kernel ... MulFunctor<float>` (~4.9%)**
   - Bound: **memory-bound** (elementwise float math)
   - Candidate: **reduce avoidable float32 elementwise work** in the dual chunk path (fewer casts / fewer intermediates), or fuse via a targeted `torch.compile` region.

## Implemented optimization (LayerNorm stats reuse) + measurements

Change (low risk):
- In the TTT dual/primal anchored paths, **avoid recomputing LayerNorm statistics twice** for the same `tK`.
- Use `torch.var_mean` (single-pass stats) and reuse `(xhat, rstd)` for the explicit LN-backward formula used to compute `u`.

Measured end-to-end throughput at `seq_len=2048` (20-step window, steps 5–24; logs not committed):
- Baseline: **645.6 toks/s avg** (range 637.8–653.7), step_s avg **25.38**
- Optimized: **712.2 toks/s avg** (range 696.8–723.1), step_s avg **23.01**
- Net: **+10.3% toks/s** (did not reach 15% end-to-end; see below)

Hotspot impact (from traces above):
- `reduce_kernel*` total GPU-kernel time: **10,238,668 µs → 7,147,513 µs** (**-30.2%**)
- `reduce_kernel*WelfordOps*` (LN variance / stats): **1,760,388 µs → 1,231,460 µs** (**-30.0%**)
- `MeanOps` reductions: **2,265,219 µs → 543,582 µs** (**-76.0%**)

Why end-to-end didn’t move 15%:
- The profile is **communication dominated** (NCCL kernels were ~57% of baseline kernel time and ~49% after the LN optimization), so further gains likely require **FSDP comm overlap / comm volume reduction**, not more LN micro-optimizations.

