from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW

from app4.ttt.model.configs import get_config
from app4.ttt.model.llama_ttt import TTTLlamaForCausalLM
from app4.ttt.train.fsdp import FSDPConfig, wrap_fsdp
from app4.ttt.utils.checkpointing import load_checkpoint, save_checkpoint
from app4.ttt.utils.logging import log_rank0, setup_logging
from app4.ttt.utils.seed import seed_all


@dataclass
class TrainArgs:
    config: str
    strategy: str
    precision: str
    device: str
    steps: int
    seq_len: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup: int
    grad_clip: float
    synthetic: bool
    save_every: int
    ckpt_dir: str
    resume: str | None
    allow_large_cpu: bool


def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_method = os.environ.get("APP4_DIST_INIT_METHOD")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if init_method:
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=int(os.environ["RANK"]),
                world_size=int(os.environ["WORLD_SIZE"]),
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
                )
            else:
                dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


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
    p.add_argument("--strategy", type=str, choices=["none", "fsdp", "deepspeed"], default="none")
    p.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--save-every", type=int, default=200)
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
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        grad_clip=args.grad_clip,
        synthetic=bool(args.synthetic),
        save_every=args.save_every,
        ckpt_dir=args.ckpt_dir,
        resume=args.resume,
        allow_large_cpu=bool(args.allow_large_cpu),
    )


def main():
    a = parse_args()
    logger = setup_logging()
    setup_dist()
    seed_all(1234)

    cfg_name = a.config
    device = torch.device(a.device)

    # Safety: avoid accidentally trying to run 1.3B+ configs on CPU in dev/CI.
    # This keeps the CLI command stable while making it runnable on CPU.
    if device.type == "cpu" and (not a.allow_large_cpu) and cfg_name != "debug_tiny":
        log_rank0(
            logger,
            f"CPU detected; config={cfg_name} is too large for CPU smoke. "
            "Switching to config=debug_tiny. Use --allow-large-cpu to disable.",
        )
        cfg_name = "debug_tiny"

    cfg = get_config(cfg_name)

    model = TTTLlamaForCausalLM(cfg).to(device=device)
    if a.precision == "bf16" and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)

    # FSDP wrap (classic FSDP only in v1)
    if a.strategy == "fsdp":
        model = wrap_fsdp(model, cfg=FSDPConfig(mixed_precision=(a.precision == "bf16")))

    optim = AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))

    start_step = 0
    if a.resume:
        start_step = load_checkpoint(a.resume, model=model, optimizer=optim, scheduler=None)
        log_rank0(logger, f"Resumed from {a.resume} at step={start_step}")

    model.train()
    t0 = time.time()
    tokens = 0

    for step in range(start_step, a.steps):
        lr_scale = cosine_with_warmup(step, warmup=a.warmup, total=a.steps)
        for pg in optim.param_groups:
            pg["lr"] = a.lr * lr_scale

        # synthetic next-token data
        input_ids = torch.randint(0, cfg.vocab_size, (a.batch_size, a.seq_len), device=device, dtype=torch.long)

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

        if a.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)

        optim.step()

        tokens += a.batch_size * a.seq_len
        if is_rank0() and (step % 10 == 0 or step == a.steps - 1):
            dt = time.time() - t0
            log_rank0(
                logger,
                f"step={step} loss={loss.item():.4f} lr={optim.param_groups[0]['lr']:.3e} toks/s={tokens/max(1e-6, dt):.1f}",
            )

        if is_rank0() and a.save_every > 0 and step > 0 and (step % a.save_every == 0):
            save_checkpoint(a.ckpt_dir, model=model, optimizer=optim, scheduler=None, step=step, cfg=cfg)

    if is_rank0():
        save_checkpoint(a.ckpt_dir, model=model, optimizer=optim, scheduler=None, step=a.steps, cfg=cfg)


if __name__ == "__main__":
    main()

