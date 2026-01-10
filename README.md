## app4 — TTT-Linear Llama-style LM (attention replaced by TTT mixer)

This repo implements **paper-faithful TTT-Linear** (Sun et al., arXiv:2407.04620) inside a Llama-style block:
- RMSNorm + residual
- SwiGLU MLP
- RoPE kept (applied to Q and K projections)
- Attention replaced by **multi-head TTT-Linear mixer** with per-head fast weights `W ∈ R^{d×d}` (d=head_dim)

### Exact inner model (paper-faithful; non-negotiable)

Inner linear model: `f_res(z; W) = W z`

Stability wrapper (MUST be exactly this):
`f(z; W) = z + LayerNorm(W z)`

**Important:** this is NOT `LayerNorm(z + Wz)`.

### Learned multi-view projections

Per layer:
- `K = θK x`
- `V = θV x`
- `Q = θQ x`

### Input-gated inner learning rate (baseline)

`eta(x) = eta_base * sigmoid(theta_lr · x_model)`

Implemented as a linear projection from model space to either a scalar or per-head gate.

### Dual / primal split + causal chunk semantics

- **Training/prefill** uses **chunked mini-batch GD** (chunk size `b_ttt=16`) via the **dual form**.
- **Decode** uses **primal sequential** updates.

Causality requirement: token `i` output reflects updates from tokens `< i` (strictly prior) within the chunk.

### Training vs Decode semantics (locked invariant)

- **Training / Prefill** (chunked, `b_ttt=16`): **anchored mini-batch GD**
  - `u_i` is computed at the **chunk-start anchor** `W_mod` (not the evolving within-chunk `W_i`).
  - Token `i` output includes updates from tokens `< i` (strict causal via strict-lower-triangular update application).
- **Decode** (autoregressive, `b=1`): **online GD**
  - `u_t` is computed at the **current** `W_t`.
  - Updates are applied sequentially, one token at a time.

These semantics are enforced in code: calling the decode path (`use_dual=False`) while the model is in `.train()` mode raises an error.

### Checkpointing THROUGH TIME (mandatory)

Naively backpropagating through the recurrent fast-weight state `W_t` stores `W_1..W_T` and explodes memory.

This repo implements **segment checkpointing through time**:
- process the sequence in segments of `checkpoint_n_chunks * b_ttt` tokens
- `torch.utils.checkpoint` wraps each segment
- only segment-boundary `W` is stored; intermediate `W` inside the segment is recomputed in backward

You can tune:
- `checkpoint_n_chunks` in config (`0` disables)

---

## Install

CPU (CI-like):

```bash
python -m pip install -U pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -e ".[dev]"
pytest
```

GPU (CUDA wheels depend on your environment; install torch accordingly):

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,triton]"
pytest
```

---

## Run unit tests (critical)

```bash
pytest
```

Includes:
- dual vs primal match (Z and W_next)
- strict chunk causality
- per-sequence state isolation
- b=1 dual matches primal step (inference semantics)

---

## Smoke train (synthetic)

```bash
python -m app4.ttt.train.train --config bringup_1p3b --steps 1 --seq-len 32 --batch-size 1 --device cpu --precision fp32 --synthetic --save-every 1 --ckpt-dir checkpoints/smoke
python -m app4.ttt.train.train --config bringup_1p3b --steps 2 --seq-len 32 --batch-size 1 --device cpu --precision fp32 --synthetic --resume checkpoints/smoke
```

---

## Supported distributed strategy (v1)

- `--strategy none`: single-process (default)
- `--strategy fsdp`: **classic FSDP only** (checkpoint-compatible sharded state_dict)

Unsupported strategies (fail fast): `deepspeed`, `fsdp2`, composable sharding.

---

## Smoke train + checkpoint round-trip (classic FSDP)

```bash
torchrun --standalone --nproc_per_node=2 -m app4.ttt.train.train --config bringup_1p3b --steps 1 --seq-len 32 --batch-size 1 --device cpu --precision fp32 --synthetic --save-every 1 --ckpt-dir checkpoints/smoke --strategy fsdp
torchrun --standalone --nproc_per_node=2 -m app4.ttt.train.train --config bringup_1p3b --steps 2 --seq-len 32 --batch-size 1 --device cpu --precision fp32 --synthetic --resume checkpoints/smoke --ckpt-dir checkpoints/smoke --strategy fsdp
```

---

## Real data training (tiny.txt + SentencePiece) — non-demo gate

```bash
python -m app4.ttt.train.train --config bringup_1p3b --steps 200 --seq-len 128 --batch-size 1 --device cpu --precision fp32 --dataset app4/ttt/data/assets/tiny.txt --tokenizer spm_train
```

Note: on CPU the train script will automatically switch large configs to `debug_tiny` unless you pass `--allow-large-cpu`.

---

## Benchmark: dual vs primal (GPU perf gate)

```bash
python -m app4.ttt.utils.bench_ttt --device cuda --dtype bf16 --batch 16 --heads 64 --head-dim 128 --chunk 16 --assert-speedup 1.5
```

---

## Inference / streaming generate (token ids)

```bash
python -m app4.ttt.inference.generate --config bringup_1p3b --prompt-tokens "1,2,3,4" --max-new-tokens 16
```

This is token-id based. (Training provides a SentencePiece bring-up tokenizer via `--tokenizer spm_train`, but inference is still token-id driven.)

