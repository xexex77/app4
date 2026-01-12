from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW

from app4.ttt.data.datasets import load_token_ids
from app4.ttt.data.sampler import ShardedTokenStream, dist_rank_world
from app4.ttt.data.tokenizer import ByteTokenizer
from app4.ttt.model.configs import get_config
from app4.ttt.model.llama_ttt import TTTLlamaForCausalLM
from app4.ttt.train.fsdp import FSDPConfig, wrap_fsdp
from app4.ttt.train.train import setup_dist  # re-use the repo's dist setup
from app4.ttt.utils.logging import log_rank0, setup_logging
from app4.ttt.utils.seed import seed_all


def _needs_torchrun(strategy: str) -> bool:
    return strategy == "fsdp" and "RANK" not in os.environ


def _relaunch_under_torchrun(argv: list[str], *, nproc_per_node: int):
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={int(nproc_per_node)}",
        "-m",
        "app4.ttt.utils.bench_trainstep",
    ] + argv
    raise SystemExit(subprocess.call(cmd))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="bringup_1p3b")
    p.add_argument("--strategy", type=str, choices=["none", "fsdp"], default="none")
    p.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=0.0)

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--tokenizer", type=str, choices=["bytes"], default="bytes")

    # FSDP comm knobs (used only when --strategy fsdp)
    p.add_argument(
        "--fsdp-wrap",
        "--fsdp_wrap",
        type=str,
        choices=["auto", "block", "block2"],
        default="auto",
    )
    p.add_argument("--fsdp-bucket-cap-mb", "--fsdp_bucket_cap_mb", type=int, default=0)
    p.add_argument(
        "--fsdp-reshard-after-forward",
        "--fsdp_reshard_after_forward",
        type=int,
        choices=[0, 1],
        default=1,
    )
    p.add_argument(
        "--fsdp-limit-all-gathers",
        "--fsdp_limit_all_gathers",
        type=int,
        choices=[0, 1],
        default=1,
    )

    # When launched via `python -m`, bench will self-launch torchrun for FSDP.
    p.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        type=int,
        default=int(torch.cuda.device_count()) if torch.cuda.is_available() else 1,
        help="Used only for self-launch under torchrun when --strategy fsdp.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    raw_argv = argv if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)

    # Self-launch torchrun for FSDP if invoked as `python -m ...`.
    if _needs_torchrun(args.strategy):
        _relaunch_under_torchrun(
            argv=raw_argv,
            nproc_per_node=int(args.nproc_per_node),
        )

    logger = setup_logging()
    device = torch.device(args.device)
    setup_dist(device=device)
    rank, world = dist_rank_world()
    seed_all(int(args.seed))

    if args.tokenizer != "bytes":
        raise ValueError("Only --tokenizer bytes is supported in this bench.")
    tokenizer = ByteTokenizer()

    base_cfg = get_config(str(args.config))
    tokens = load_token_ids(args.dataset, tokenizer=tokenizer)
    token_stream = ShardedTokenStream(
        tokens,
        seq_len=int(args.seq_len),
        batch_size=int(args.batch_size),
        rank=rank,
        world_size=world,
    )

    cfg = base_cfg
    if tokenizer.vocab_size > base_cfg.vocab_size:
        cfg = get_config(str(args.config), vocab_size=tokenizer.vocab_size)

    model = TTTLlamaForCausalLM(cfg)
    if args.precision == "bf16" and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    model = model.to(device=device)

    if args.strategy == "fsdp":
        bucket_cap_mb = int(args.fsdp_bucket_cap_mb)
        fsdp_cfg = FSDPConfig(
            mixed_precision=(args.precision == "bf16"),
            wrap=str(args.fsdp_wrap),
            bucket_cap_mb=(bucket_cap_mb if bucket_cap_mb > 0 else None),
            reshard_after_forward=bool(int(args.fsdp_reshard_after_forward)),
            limit_all_gathers=bool(int(args.fsdp_limit_all_gathers)),
        )
        log_rank0(
            logger,
            "FSDP config: "
            f"wrap={fsdp_cfg.wrap} reshard_after_forward={int(fsdp_cfg.reshard_after_forward)} "
            f"limit_all_gathers={int(bool(fsdp_cfg.limit_all_gathers))} "
            f"bucket_cap_mb={fsdp_cfg.bucket_cap_mb}",
        )
        model = wrap_fsdp(model, cfg=fsdp_cfg)

    optim = AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (args.precision == "bf16" and device.type == "cuda")
        else nullcontext()
    )

    warmup = int(args.warmup)
    steps = int(args.steps)
    if steps <= warmup + 1:
        raise ValueError("--steps must be > --warmup + 1 for a meaningful window.")
    window_start = warmup
    window_end = warmup + 19  # inclusive (20 steps)
    if steps <= window_end:
        raise ValueError("--steps must be >= warmup+20 (e.g., 30 with warmup=5).")

    toks_per_step_global = int(args.batch_size) * int(args.seq_len) * int(world)

    step_s_vals: list[float] = []
    toks_s_vals: list[float] = []
    peak_mem_gb_max = 0.0

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)

    for step in range(steps):
        input_ids = token_stream.next_batch().to(device=device)

        t0 = time.perf_counter()
        with autocast_ctx:
            logits, _ = model(input_ids, cache=None, use_dual=True, checkpoint_ttt=True)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, cfg.vocab_size),
                input_ids[:, 1:].reshape(-1),
            )

        optim.zero_grad(set_to_none=True)
        loss.backward()

        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optim.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        dt = time.perf_counter() - t0

        if window_start <= step <= window_end and rank == 0:
            step_s_vals.append(float(dt))
            toks_s_vals.append(float(toks_per_step_global) / max(1e-9, float(dt)))
            if device.type == "cuda":
                peak_mem_gb_max = max(
                    peak_mem_gb_max, float(torch.cuda.max_memory_allocated(device=device)) / 1e9
                )

        if rank == 0 and (step in {0, warmup, window_end}):
            log_rank0(logger, f"bench step={step} step_s={dt:.3f}")

    if rank == 0:
        toks_avg = sum(toks_s_vals) / max(1, len(toks_s_vals))
        step_avg = sum(step_s_vals) / max(1, len(step_s_vals))
        log_rank0(
            logger,
            "bench_result "
            f"toks_s_avg={toks_avg:.2f} step_s_avg={step_avg:.3f} "
            f"peak_mem_gb_max={peak_mem_gb_max:.2f} "
            f"window=[{window_start},{window_end}] world={world}",
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":  # pragma: no cover
    main()

