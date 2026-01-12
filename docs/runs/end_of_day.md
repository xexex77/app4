# End-of-day status

Date: 2026-01-12 (UTC)

- **Git SHA**: `8cc0958a69206756f8ddb2b78b54d14faf8a74b3`
- **Run**: `ttt_llama_47b` — `--strategy fsdp --precision bf16` — `seq_len=2048` — `batch_size=1` — `grad_accum_steps=1`
- **Dataset**: `/home/ubuntu/datasets/tinyshakespeare.txt` (bytes tokenizer)

## Training status

- **Last completed step**: `1000`
- **Checkpoint**:
  - Base: `/home/ubuntu/checkpoints/long_2048_accum1`
  - Latest: `/home/ubuntu/checkpoints/long_2048_accum1/step_000001000`
  - Manifest: `/home/ubuntu/checkpoints/long_2048_accum1/checkpoint_manifest.json`
- **Log**: `/home/ubuntu/runs/logs/long_2048_accum1.log`

## Performance (near end)

- **toks/s range**:
  - Steps `900–990`: ~`678–685` toks/s
  - Step `1000`: `553` toks/s (slower boundary step)
- **peak_mem_gb**: `86.52`

## Shutdown + resume sanity

- **Shutdown**: waited for `checkpoint saved: step=1000`, then sent a single `Ctrl-C` to the `screen` train session; verified no remaining `torchrun` processes.
- **2-step resume sanity**: **not run** (skipped).

