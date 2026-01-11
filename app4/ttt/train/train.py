from __future__ import annotations

import argparse
import os
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW

from app4.ttt.data.datasets import load_token_ids
from app4.ttt.data.sampler import ShardedTokenStream, dist_rank_world
from app4.ttt.data.tokenizer import (
    ByteTokenizer,
    SentencePieceTokenizer,
    SentencePieceTrainConfig,
    train_sentencepiece,
)
from app4.ttt.model.configs import get_config
from app4.ttt.model.llama_ttt import TTTLlamaForCausalLM
from app4.ttt.train.fsdp import FSDPConfig, wrap_fsdp
from app4.ttt.utils.checkpointing import load_checkpoint, save_checkpoint
from app4.ttt.utils.logging import log_rank0, setup_logging
from app4.ttt.utils.rng import get_rng_state, set_rng_state
from app4.ttt.utils.seed import seed_all


@dataclass
class TrainArgs:
    config: str
    strategy: str
    precision: str
    device: str
    seed: int
    steps: int
    seq_len: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup: int
    grad_clip: float
    synthetic: bool
    dataset: str | None
    tokenizer: str | None
    tokenizer_model: str | None
    tokenizer_vocab_size: int | None
    save_every: int
    no_save: bool
    log_every: int
    ckpt_dir: str
    resume: str | None
    allow_large_cpu: bool


def setup_dist(*, device: torch.device | None = None):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        init_method = os.environ.get("APP4_DIST_INIT_METHOD")
        if device is not None:
            # IMPORTANT: backend must follow the requested device, not mere CUDA availability.
            # Step-7 gate runs FSDP under `torchrun ... --device cpu`, which should use gloo even
            # on GPU machines.
            if device.type == "cpu":
                backend = "gloo"
            elif device.type == "cuda":
                backend = "nccl"
            else:
                backend = "gloo"
        else:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        device_id = None
        if device is not None and device.type == "cuda":
            # IMPORTANT: set the local CUDA device *before* initializing NCCL, and pass
            # device_id to init_process_group to avoid barrier/device mapping warnings.
            torch.cuda.set_device(local_rank)
            device_id = local_rank

        if init_method:
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=int(os.environ["RANK"]),
                world_size=int(os.environ["WORLD_SIZE"]),
                device_id=device_id,
            )
        else:
            # On some Windows wheels, env:// rendezvous uses TCPStore(use_libuv=True) and fails
            # when libuv isn't compiled. For local dev on Windows, fall back to file:// init.
            if os.name == "nt":
                rank = int(os.environ["RANK"])
                world = int(os.environ["WORLD_SIZE"])
                port = os.environ.get("MASTER_PORT", "29500")
                tmp = Path(tempfile.gettempdir())
                init_file = tmp / f"app4_pg_{port}.txt"
                if rank == 0:
                    init_file.write_text("app4")
                dist.init_process_group(
                    backend=backend,
                    init_method=f"file:///{str(init_file).replace(os.sep, '/')}",
                    rank=rank,
                    world_size=world,
                    device_id=device_id,
                )
            else:
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    device_id=device_id,
                )


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def cosine_with_warmup(step: int, *, warmup: int, total: int):
    if step < warmup:
        return float(step) / float(max(1, warmup))
    progress = (step - warmup) / float(max(1, total - warmup))
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="bringup_1p3b")
    # DeepSpeed is intentionally unsupported in v1 (must be real or removed).
    p.add_argument("--strategy", type=str, choices=["none", "fsdp"], default="none")
    p.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a real text dataset (e.g. tiny.txt).",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        choices=["bytes", "spm_train", "spm_model"],
        default=None,
        help="Tokenizer backend (required when --dataset is set).",
    )
    p.add_argument(
        "--tokenizer-model",
        type=str,
        default=None,
        help="Path to a SentencePiece .model (spm_model).",
    )
    p.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        default=None,
        help=(
            "Requested SentencePiece vocab size when using spm_train "
            "(defaults to config vocab_size)."
        ),
    )
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Disable all training checkpoint writes (including the final checkpoint).",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/run1")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--allow-large-cpu",
        action="store_true",
        help="Allow running very large configs on CPU (may OOM / be extremely slow).",
    )
    args = p.parse_args()

    return TrainArgs(
        config=args.config,
        strategy=args.strategy,
        precision=args.precision,
        device=args.device,
        seed=int(args.seed),
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        grad_clip=args.grad_clip,
        synthetic=bool(args.synthetic),
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        tokenizer_model=args.tokenizer_model,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        save_every=args.save_every,
        no_save=bool(args.no_save),
        log_every=int(args.log_every),
        ckpt_dir=args.ckpt_dir,
        resume=args.resume,
        allow_large_cpu=bool(args.allow_large_cpu),
    )


def main():
    a = parse_args()
    device = torch.device(a.device)
    logger = setup_logging()
    setup_dist(device=device)
    rank, world = dist_rank_world()
    seed_all(int(a.seed))

    if a.synthetic and a.dataset:
        raise ValueError("--synthetic and --dataset are mutually exclusive.")
    if (a.dataset is not None) and (a.tokenizer is None):
        raise ValueError("--tokenizer is required when using --dataset.")

    cfg_name = a.config

    # Safety: avoid accidentally trying to run 1.3B+ configs on CPU in dev/CI.
    # This keeps the CLI command stable while making it runnable on CPU.
    if device.type == "cpu" and (not a.allow_large_cpu) and cfg_name != "debug_tiny":
        log_rank0(
            logger,
            f"CPU detected; config={cfg_name} is too large for CPU smoke. "
            "Switching to config=debug_tiny. Use --allow-large-cpu to disable.",
        )
        cfg_name = "debug_tiny"

    base_cfg = get_config(cfg_name)

    # Real data pipeline (non-demo gate): tokenizer + deterministic token stream.
    tokenizer = None
    token_stream = None
    cfg = base_cfg
    if a.dataset is not None:
        ckpt_dir = Path(a.ckpt_dir)
        tok_dir = ckpt_dir / "tokenizer"
        tok_dir.mkdir(parents=True, exist_ok=True)

        if a.tokenizer == "bytes":
            tokenizer = ByteTokenizer()
        elif a.tokenizer == "spm_train":
            model_prefix = tok_dir / "spm"
            model_file = model_prefix.with_suffix(".model")
            if (rank == 0) and (not model_file.exists()):
                req_vocab = int(a.tokenizer_vocab_size or base_cfg.vocab_size)
                train_sentencepiece(
                    input_path=a.dataset,
                    model_prefix=model_prefix,
                    cfg=SentencePieceTrainConfig(vocab_size=req_vocab, seed=int(a.seed)),
                )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            tokenizer = SentencePieceTokenizer(model_file)
        elif a.tokenizer == "spm_model":
            if not a.tokenizer_model:
                raise ValueError("--tokenizer-model is required for --tokenizer spm_model.")
            tokenizer = SentencePieceTokenizer(a.tokenizer_model)
        else:
            raise ValueError(f"Unsupported tokenizer: {a.tokenizer!r}")

        # Keep the model config vocab size unless the tokenizer requires a larger vocab.
        # (This lets us do real-data bring-up with small tokenizers while preserving the
        # parameter count of large configs like ttt_llama_47b.)
        if tokenizer.vocab_size > base_cfg.vocab_size:
            cfg = get_config(cfg_name, vocab_size=tokenizer.vocab_size)

        tokens = load_token_ids(a.dataset, tokenizer=tokenizer)
        token_stream = ShardedTokenStream(
            tokens, seq_len=a.seq_len, batch_size=a.batch_size, rank=rank, world_size=world
        )

    model = TTTLlamaForCausalLM(cfg)
    if a.precision == "bf16" and device.type == "cuda":
        # Avoid materializing fp32 weights on GPU first (can OOM for very large configs).
        model = model.to(dtype=torch.bfloat16)
    model = model.to(device=device)

    # FSDP wrap (classic FSDP only in v1)
    if a.strategy == "fsdp":
        model = wrap_fsdp(model, cfg=FSDPConfig(mixed_precision=(a.precision == "bf16")))

    optim = AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))

    start_step = 0
    if a.resume:
        start_step, meta, rng_state = load_checkpoint(
            a.resume, model=model, optimizer=optim, scheduler=None
        )
        if rng_state is not None:
            set_rng_state(rng_state)
        if token_stream is not None:
            data_state = (meta.get("extra") or {}).get("data_state", None)
            if isinstance(data_state, dict):
                token_stream.load_state_dict(data_state)
        log_rank0(logger, f"Resumed from {a.resume} at step={start_step}")

    model.train()
    tokens = 0
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    tokens_per_step_global = a.batch_size * a.seq_len * world_size

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)
    interval_t0 = time.perf_counter()
    interval_steps = 0

    if is_rank0() and a.no_save:
        log_rank0(logger, "Checkpoint saving disabled (--no-save).")

    for step in range(start_step, a.steps):
        lr_scale = cosine_with_warmup(step, warmup=a.warmup, total=a.steps)
        for pg in optim.param_groups:
            pg["lr"] = a.lr * lr_scale

        if a.dataset is not None:
            assert token_stream is not None
            input_ids = token_stream.next_batch().to(device=device)
        else:
            # synthetic next-token data
            input_ids = torch.randint(
                0, cfg.vocab_size, (a.batch_size, a.seq_len), device=device, dtype=torch.long
            )

        # forward (grad always enabled; autocast only for CUDA bf16)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (a.precision == "bf16" and device.type == "cuda")
            else nullcontext()
        )
        with autocast_ctx:
            logits, _ = model(input_ids, cache=None, use_dual=True, checkpoint_ttt=True)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, cfg.vocab_size),
                input_ids[:, 1:].reshape(-1),
            )

        optim.zero_grad(set_to_none=True)
        loss.backward()

        if not torch.isfinite(loss.detach()):
            raise RuntimeError(f"Non-finite loss at step={step}: {loss.detach().float().item()}")

        grad_norm = None
        if a.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)

        optim.step()
        interval_steps += 1

        tokens += a.batch_size * a.seq_len
        do_log = (
            a.log_every > 0
            and (step % a.log_every == 0 or step == a.steps - 1)
            and is_rank0()
        )
        if do_log:
            if device.type == "cuda":
                torch.cuda.synchronize(device=device)
            dt = time.perf_counter() - interval_t0
            step_s = dt / max(1, interval_steps)
            toks_per_s = float(tokens_per_step_global * interval_steps) / max(1e-9, dt)

            peak_mem_gb = 0.0
            if device.type == "cuda":
                peak_mem_gb = float(torch.cuda.max_memory_allocated(device=device)) / 1e9

            gn = float("nan")
            if grad_norm is not None:
                try:
                    gn = float(grad_norm)
                except TypeError:
                    gn = float(grad_norm.item())

            lr = float(optim.param_groups[0]["lr"])
            log_rank0(
                logger,
                f"step={step} loss={loss.detach().float().item():.4f} grad_norm={gn:.3e} "
                f"step_s={step_s:.3f} toks/s={toks_per_s:.1f} "
                f"peak_mem_gb={peak_mem_gb:.2f} lr={lr:.3e}",
            )

            interval_t0 = time.perf_counter()
            interval_steps = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device=device)

        # NOTE: under FSDP, state_dict collection uses collective comms; all ranks must
        # participate even if only rank0 writes metadata.
        if (
            (not a.no_save)
            and a.save_every > 0
            and (step + 1) < a.steps
            and ((step + 1) % a.save_every == 0)
        ):
            extra = {"data_state": token_stream.state_dict() if token_stream is not None else None}
            if is_rank0():
                log_rank0(logger, f"saving checkpoint: step={step+1} dir={a.ckpt_dir}")
            if device.type == "cuda":
                torch.cuda.synchronize(device=device)
            ckpt_t0 = time.perf_counter()
            save_checkpoint(
                a.ckpt_dir,
                model=model,
                optimizer=optim,
                scheduler=None,
                step=step + 1,
                cfg=cfg,
                extra=extra,
                rng_state=get_rng_state(),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device=device)
            if is_rank0():
                dt_s = time.perf_counter() - ckpt_t0
                log_rank0(logger, f"checkpoint saved: step={step+1} dt_s={dt_s:.1f}")

    if not a.no_save:
        extra = {"data_state": token_stream.state_dict() if token_stream is not None else None}
        if is_rank0():
            log_rank0(logger, f"saving final checkpoint: step={a.steps} dir={a.ckpt_dir}")
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        ckpt_t0 = time.perf_counter()
        save_checkpoint(
            a.ckpt_dir,
            model=model,
            optimizer=optim,
            scheduler=None,
            step=a.steps,
            cfg=cfg,
            extra=extra,
            rng_state=get_rng_state(),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        if is_rank0():
            dt_s = time.perf_counter() - ckpt_t0
            log_rank0(logger, f"final checkpoint saved: step={a.steps} dt_s={dt_s:.1f}")

    # Clean shutdown to avoid NCCL/Gloo resource leak warnings.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

