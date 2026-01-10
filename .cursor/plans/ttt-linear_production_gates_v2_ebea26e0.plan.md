---
name: TTT-Linear production gates v2
overview: "Augment the hardening plan with non-demo requirements: a minimal real data pipeline with deterministic sharding/resume, explicit anchored-vs-online semantics invariants, a dual-vs-primal performance microbenchmark gate, and strict supported-distributed policy (classic FSDP-only v1; DeepSpeed optional but must not lie)."
todos:
  - id: materialize-skeleton
    content: Write the pasted skeleton into the empty repo with packaging/tests/CI wired.
    status: pending
  - id: learned-w0
    content: Add learned per-head W0 and use it for init_state/reset_cache; prevent shared-storage state across batch.
    status: pending
    dependencies:
      - materialize-skeleton
  - id: fp32-fastweights
    content: Store fast-weight cache state in fp32 and accumulate updates in fp32; define compute casts explicitly.
    status: pending
    dependencies:
      - learned-w0
  - id: grad-mean-b
    content: Scale inner-loss gradient by chunk size b (mean reduction default) for chunked training semantics.
    status: pending
    dependencies:
      - fp32-fastweights
  - id: semantics-invariants
    content: Document and enforce anchored-MBGD training vs online-GD decode invariants in docs/tests/code.
    status: pending
    dependencies:
      - grad-mean-b
  - id: tests-expand
    content: Add tests for W0 semantics, fp32 state, and GPU bf16 dual/primal tolerance.
    status: pending
    dependencies:
      - semantics-invariants
  - id: train-autocast-fix
    content: Fix training autocast/grad context (no placeholders), correct step accounting.
    status: pending
    dependencies:
      - grad-mean-b
  - id: fsdp-policy
    content: "Classic FSDP-only v1: remove composable fully_shard path; make checkpointing/resume compatible and fail-fast for unsupported strategies."
    status: pending
    dependencies:
      - train-autocast-fix
  - id: real-data-pipeline
    content: Implement tokenizer hook + minimal dataset + deterministic sharding/resume; add tiny real dataset asset; wire into train entrypoint.
    status: pending
    dependencies:
      - train-autocast-fix
  - id: bench-dual-vs-primal
    content: Add microbenchmark comparing dual vs primal throughput/memory at H=64,d=128,b=16; require dual faster.
    status: pending
    dependencies:
      - grad-mean-b
  - id: deepspeed-optional
    content: "DeepSpeed ZeRO-3 is optional: either implement real DS engine+checkpoint/resume or remove the flag and scripts."
    status: pending
    dependencies:
      - train-autocast-fix
  - id: smoke-save-load
    content: One-step (and resume) smoke train that saves/loads model+optimizer+data iterator+RNG under supported sharding strategy.
    status: pending
    dependencies:
      - fsdp-policy
      - real-data-pipeline
  - id: docs-ci-enforce
    content: Update README/scripts/CI; enforce supported distributed options (fail fast), include real-data training example.
    status: pending
    dependencies:
      - smoke-save-load
---

# TTT-Linear Repo: Production-Ready v1 Plan (with non-demo gates)

## Goals

- Ship a **correct, resumable, and scalable** TTT-Linear implementation (paper-faithful) that can be brought up at ~1.3B and then scaled to ~47B on 8×B200.
- Prevent “demo-ware” by including a **real data path**, **explicit training semantics invariants**, and a **performance gate**.

## Non-negotiable requirements (must-fix before merge)

1) **Learned fast-weight init `W0` per layer/head**; `init_state` and `reset_cache` use `W0` (no zero-init baseline).
2) **Fast weights `W` state and updates in fp32** even when model runs bf16; clearly define cast points.
3) **Distributed truthfulness**: v1 supports **classic FSDP only** *or* implements real FSDP2 checkpointing via `torch.distributed.checkpoint`. (Plan chooses classic FSDP for v1.)
4) **DeepSpeed must be real or removed**. In v1 it is **optional**, but leaving “it runs but lies” is forbidden.
5) **Training autocast cleanup**: no placeholder/no_grad confusion; grad always enabled.
6) **Inner-loss gradient scaled by chunk size `b`** (mean reduction default).
7) **Real data pipeline** (tokenizer + dataset + deterministic sharding + resume). Synthetic-only is not acceptable.
8) **Semantics invariants locked**: training/prefill uses anchored mini-batch GD; decode uses online/primal.
9) **Performance gate**: dual path must be meaningfully faster than primal at `b=16, d=128, H=64`.
10) **Unsupported distributed flags fail fast** (CI/README/scripts refuse unimplemented options).

## Semantics invariants (written + enforced)

- **Training/prefill (chunked, `b_ttt=16`)**:
- **Anchored MBGD**: `u_i` is computed using `W_mod` (chunk-start anchor), not the evolving `W_i`.
- **Causal application**: token `i` output reflects updates from tokens `< i` within the chunk.
- Implementation: dual form + anchored primal reference for testing.
- **Decode (autoregressive, `b=1`)**:
- **Online GD**: `u_t` is computed at the current `W_t`.
- Sequential primal update.

## Fast-weight precision policy (written invariant)

- Cache/state `W_state` is stored as **fp32**.
- For compute:
- `W_compute = W_state.to(dtype=x.dtype)` (bf16 under autocast) for matmuls producing `tK/tQ`.
- `dW` accumulation uses fp32: `(U_eta.float().T @ K.float())`.
- Update: `W_state = W_state - dW_fp32`.

## Numbered task list (dependencies + acceptance criteria)

### 0) Materialize the skeleton into the empty repo

- **Depends on**: none.
- **Work**: create the file tree (packaging, model, tests, scripts, CI) from the pasted skeleton as baseline.
- **Acceptance**:
- `python -m pip install -e ".[dev]"` succeeds.
- `pytest` runs (may fail until later tasks, but imports must work).

### 1) Add learned `W0` init and correct reset semantics

- **Depends on**: Task 0.
- **Files**: `app4/ttt/layers/ttt_linear.py`, `app4/ttt/model/llama_ttt.py`.
- **Work**:
- Add `self.W0 = nn.Parameter` with shape `(H,d,d)`.
- `init_state(B)` returns `W0[None,...].expand(B,...)` **then clone/contiguous** to avoid shared storage.
- `reset_cache` copies `W0` back into each layer state.
- **Acceptance**:
- New test: `test_w0_used_for_init_and_reset` passes.
- `test_state_isolation` passes.

### 2) Enforce fp32 fast-weight state + fp32 updates (bf16-friendly compute)

- **Depends on**: Task 1.
- **Files**: `app4/ttt/layers/ttt_linear.py`, `app4/ttt/model/llama_ttt.py`.
- **Work**:
- Cache `W` always fp32.
- Cast to activation dtype only for matmul; accumulate updates in fp32.
- Ensure forward returns activations in compute dtype, but returns `W_next` fp32.
- **Acceptance**:
- New test: `test_fast_weights_fp32_state` passes.
- `pytest` passes on CPU.

### 3) Mean-reduce inner-loss gradient by chunk size `b`

- **Depends on**: Task 2.
- **Files**: `app4/ttt/layers/ttt_linear.py`.
- **Work**:
- Change `g` scaling in chunk codepaths to `(2.0 / b) * (y - V)`.
- Keep decode step semantics consistent for `b=1`.
- **Acceptance**:
- Dual vs anchored-primal tests still pass (fp32 tight tol).

### 4) Lock semantics invariants in docs + code (fail fast if violated)

- **Depends on**: Task 3.
- **Files**: `README.md`, `app4/ttt/layers/ttt_linear.py`, `app4/ttt/model/llama_ttt.py`, tests.
- **Work**:
- Add explicit documentation section “Training vs Decode semantics”.
- Rename/annotate reference functions to prevent accidental substitution (e.g., keep `ttt_primal_chunk_anchored` as the only reference).
- Add runtime assertions (debug-level or behind a flag) that:
- prefill/training uses `use_dual=True`
- decode/generate uses `use_dual=False`
- **Acceptance**:
- Dual==primal correctness test explicitly references anchored primal (docstring + test name).
- `pytest` passes.

### 5) Expand correctness tests (CPU + GPU bf16)

- **Depends on**: Task 4.
- **Files**: `app4/tests/*`.
- **Work**:
- Add GPU-only bf16 dual/primal match test with looser tolerances and clear skip if CUDA absent.
- Keep CPU fp32 test tight.
- **Acceptance**:
- CPU: `pytest` passes.
- GPU: `pytest -k dual_matches_primal` passes for fp32 and bf16.

### 6) Training entrypoint cleanup (autocast, grad, step accounting)

- **Depends on**: Task 3.
- **Files**: `app4/ttt/train/train.py`.
- **Work**:
- Replace placeholder autocast logic with a single correct pattern:
- grad always enabled
- autocast only for CUDA bf16
- Ensure `--steps` and resume step accounting are consistent.
- **Acceptance**:
- `python -m app4.ttt.train.train --config bringup_1p3b --synthetic --steps 1 --seq-len 32 --batch-size 1 --device cpu --precision fp32` runs.

### 7) Distributed policy v1: classic FSDP only, checkpoint-compatible

- **Depends on**: Task 6.
- **Files**: `app4/ttt/train/fsdp2.py` (rename/repurpose), `app4/ttt/utils/checkpointing.py`, `app4/ttt/train/train.py`, scripts.
- **Work**:
- Remove/disable composable `fully_shard` path.
- Make CLI accept `--strategy fsdp` (classic) and error on `fsdp2` unless implemented.
- Ensure save/load works under classic FSDP sharded state_dict.
- Add barriers around checkpoint operations.
- **Acceptance**:
- `torchrun --standalone --nproc_per_node=2 -m app4.ttt.train.train --config bringup_1p3b --synthetic --steps 1 --seq-len 32 --batch-size 1 --device cpu --precision fp32 --strategy fsdp` completes.
- Save/load round-trip works under FSDP (see Task 11).

### 8) Real data path (tokenizer + dataset + deterministic sharding + resume)

- **Depends on**: Task 6.
- **Files** (new):
- `app4/ttt/data/tokenizer.py` (interface + implementations)
- `app4/ttt/data/datasets.py` (text/JSONL minimal)
- `app4/ttt/data/sampler.py` (deterministic sharding + resume state)
- `app4/ttt/utils/rng.py` (save/restore RNG)
- `app4/ttt/train/train.py` (wire CLI)
- Add tiny dataset under `app4/ttt/data/assets/tiny.txt` (checked into repo)
- **Work**:
- Tokenizer hook:
- Provide a stable `Tokenizer` protocol.
- Implement at least one real tokenizer backend:
- **SentencePiece** path (optional dependency) that can load a `.model`, plus an option to train a tiny model from `tiny.txt` for bring-up.
- Optional HF wrapper (optional dependency) behind a flag.
- Dataset:
- Implement minimal streaming dataset over `tiny.txt` and/or JSONL.
- Deterministic sharding by `(rank, world_size)`.
- Save dataloader position/state (sample index / file offset), plus RNG state, into checkpoints.
- **Acceptance** (required “non-demo” gate):
- Single-proc CPU:
- `python -m app4.ttt.train.train --config bringup_1p3b --steps 200 --seq-len 128 --batch-size 1 --device cpu --precision fp32 --dataset app4/ttt/data/assets/tiny.txt --tokenizer spm_train` runs.
- Loss decreases: mean(loss[0:20]) > mean(loss[180:200]) by a meaningful margin (document threshold).
- Resume determinism:
- Run 100 steps, save, resume to 200; step count continues and data iterator state does not reset.

### 9) Performance gate: dual vs primal microbenchmark

- **Depends on**: Task 3.
- **Files** (new): `app4/ttt/utils/bench_ttt.py` (or `scripts/bench_ttt.py`), README.
- **Work**:
- Microbenchmark dual chunk vs primal sequential on representative shapes:
- `H=64, d=128, b=16` (and maybe vary T)
- report tokens/sec and peak memory (`torch.cuda.max_memory_allocated`)
- Keep it deterministic and warm up kernels.
- **Acceptance**:
- On GPU: dual must be faster than primal for `b=16` by a meaningful margin (e.g. ≥1.5×); print both numbers.
- (Not run in CI.)

### 10) DeepSpeed ZeRO-3 is optional (but must not lie)

- **Depends on**: Task 6.
- **Files**: `app4/ttt/train/deepspeed_zero3.py`, `app4/ttt/train/train.py`, `app4/scripts/launch_ds.sh`.
- **Work** (choose one branch):
- **Branch A (preferred for v1 speed)**: remove `--strategy deepspeed` and delete DS script until fully validated.
- **Branch B**: implement real DS engine init + backward/step + checkpoint/resume.
- **Acceptance**:
- Branch A: passing `--strategy deepspeed` errors loudly; no DS scripts remain.
- Branch B: DS run completes and DS checkpoint resumes successfully.

### 11) Smoke train + checkpoint round-trip under supported strategy

- **Depends on**: Tasks 7 and 8.
- **Files**: `app4/ttt/train/train.py`, `app4/ttt/utils/checkpointing.py`.
- **Work**:
- Validate save/load for model + optimizer + scheduler (if present) + data iterator state + RNG state.
- **Acceptance** (required):
- `pytest` passes on CPU.
- CPU single-proc:
- `python -m app4.ttt.train.train --config bringup_1p3b --steps 1 --seq-len 32 --batch-size 1 --device cpu --precision fp32 --synthetic --save-every 1 --ckpt-dir checkpoints/smoke`
- `python -m app4.ttt.train.train --config bringup_1p3b --steps 2 --seq-len 32 --batch-size 1 --device cpu --precision fp32 --synthetic --resume checkpoints/smoke`
- Classic FSDP (CPU or GPU): same round-trip under `torchrun` with `--strategy fsdp`.

### 12) Docs/scripts/CI: enforce supported options, refuse unsupported

- **Depends on**: Tasks 7–11.
- **Files**: `README.md`, `app4/scripts/*`, `.github/workflows/ci.yml`.
- **Work**:
- Ensure README uses real-data example + synthetic debug example.
- Ensure scripts only reference supported strategies.
- Ensure CLI errors loudly for unsupported distributed strategies.
- **Acceptance**:
- CI passes: `ruff check .` + `pytest` (CPU).
- Running an unsupported strategy flag fails fast with a clear error.

## Must-fix before merge (summary) + why

- **Learned `W0`**: paper-faithful stability prior; zero-init breaks your explicit requirement.
- **fp32 fast-weight state/updates**: bf16 state update drift is a common failure mode at scale.
- **Mean reduction by `b`**: avoids gradient scaling with chunk size and stabilizes tuning.
- **Autocast cleanup**: training loop correctness must be boring.
- **Classic FSDP-only v1 (or real FSDP2+DCP)**: prevents broken resume from mismatched sharding/checkpoint APIs.
- **DeepSpeed must not lie**: either validated ZeRO-3 or removed.
- **Real data pipeline + deterministic resume**: “production-ready” requires more than synthetic tokens; ensures end-to-end correctness and resumability.
- **Semantics invariants documented/enforced**: prevents accidental mixing of MBGD vs online GD semantics.
- **Performance microbenchmark**: ensures dual form is actually buying speed before renting B200 time.
- **Fail-fast unsupported flags**: prevents silent misconfiguration.

## Non-blocking follow-ups (post-v1)

- Add `torch.distributed.checkpoint` path and composable FSDP2 support.
- Implement Triton fused dual kernel in `app4/ttt/kernels/dual_triton.py` once correctness is locked.
- Expand datasets: WebDataset/HF datasets streaming; mid-epoch resume at shard granularity.
- Add tokenizer download/caching policy for offline clusters.