from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
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
    grad_accum_steps: int
    synthetic: bool
    dataset: str | None
    tokenizer: str | None
    tokenizer_model: str | None
    tokenizer_vocab_size: int | None
    save_every: int
    save_optimizer: bool
    max_checkpoints: int
    no_save: bool
    log_every: int
    ckpt_dir: str
    resume: str | None
    allow_large_cpu: bool
    fsdp_wrap: str
    fsdp_bucket_cap_mb: int
    fsdp_reshard_after_forward: int
    fsdp_limit_all_gathers: int
    eval_only: bool
    eval_steps: int
    eval_split: float
    eval_every: int
    profile_steps: int
    profile_dir: str | None


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


def _repo_root() -> Path:
    # .../app4/app4/ttt/train/train.py -> repo root is .../app4
    return Path(__file__).resolve().parents[3]


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_repo_root()))
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return "unknown"


def _list_step_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    if not ckpt_dir.exists():
        return []
    out: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        if p.is_dir() and p.name.startswith("step_") and p.name[5:].isdigit():
            out.append((int(p.name[5:]), p))
    out.sort(key=lambda x: x[0])
    return out


def _latest_checkpoint_dir(ckpt_dir: Path) -> Path | None:
    ckpts = _list_step_checkpoints(ckpt_dir)
    if ckpts:
        return ckpts[-1][1]
    # Back-compat: old layout wrote meta.json directly under ckpt_dir.
    if (ckpt_dir / "meta.json").exists():
        return ckpt_dir
    return None


def _resolve_resume_dir(*, resume: str | None, ckpt_dir: Path) -> Path | None:
    if resume is None:
        return None
    norm = resume.strip().lower()
    if norm in {"", "none", "false", "0"}:
        return None
    if norm == "auto":
        manifest = ckpt_dir / "checkpoint_manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text())
                rel = data.get("checkpoint_dir", None)
                if isinstance(rel, str) and rel:
                    p = ckpt_dir / rel
                    if p.is_dir() and (p / "meta.json").exists():
                        return p
            except Exception:
                pass
        return _latest_checkpoint_dir(ckpt_dir)

    p = Path(resume)
    if p.is_dir():
        if (p / "meta.json").exists():
            return p
        return _latest_checkpoint_dir(p)
    raise ValueError(f"--resume path does not exist or is not a directory: {resume!r}")


def _write_checkpoint_manifest(
    *,
    ckpt_dir: Path,
    checkpoint_dir: Path,
    step: int,
    cfg_name: str,
    tokenizer: str | None,
    dataset: str | None,
    world_size: int,
    seed: int,
    save_optimizer: bool,
):
    rel = os.path.relpath(str(checkpoint_dir), start=str(ckpt_dir))
    payload = {
        "format": 1,
        "step": int(step),
        "checkpoint_dir": rel,
        "config": cfg_name,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "git_sha": _git_sha(),
        "world_size": int(world_size),
        "seed": int(seed),
        "save_optimizer": int(bool(save_optimizer)),
    }
    (ckpt_dir / "checkpoint_manifest.json").write_text(json.dumps(payload, indent=2))


def _prune_checkpoints(*, ckpt_dir: Path, max_checkpoints: int, logger):
    if max_checkpoints <= 0:
        return
    ckpts = _list_step_checkpoints(ckpt_dir)
    if len(ckpts) <= max_checkpoints:
        return
    to_delete = ckpts[: len(ckpts) - max_checkpoints]
    for step, p in to_delete:
        log_rank0(logger, f"pruning old checkpoint: step={step} dir={p}")
        shutil.rmtree(p, ignore_errors=True)


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
    p.add_argument(
        "--grad-accum-steps",
        "--grad_accum_steps",
        type=int,
        default=1,
        help=(
            "Gradient accumulation steps per optimizer step. For FSDP, uses no_sync() for "
            "microsteps < N."
        ),
    )
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
        choices=["bytes", "spm_train", "spm_load", "spm_model"],
        default=None,
        help="Tokenizer backend (required when --dataset is set).",
    )
    p.add_argument(
        "--tokenizer-model",
        type=str,
        default=None,
        help="Path to a SentencePiece .model (spm_load).",
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
    p.add_argument("--save_optimizer", "--save-optimizer", type=int, choices=[0, 1], default=1)
    p.add_argument("--max_checkpoints", "--max-checkpoints", type=int, default=3)
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Disable all training checkpoint writes (including the final checkpoint).",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/run1")
    p.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="Checkpoint resume mode: 'auto' (default), 'none', or a checkpoint directory path.",
    )
    p.add_argument(
        "--allow-large-cpu",
        action="store_true",
        help="Allow running very large configs on CPU (may OOM / be extremely slow).",
    )
    p.add_argument(
        "--fsdp-wrap",
        "--fsdp_wrap",
        type=str,
        choices=["auto", "block", "block2"],
        default="auto",
        help="FSDP wrapping policy (only used when --strategy fsdp).",
    )
    p.add_argument(
        "--fsdp-bucket-cap-mb",
        "--fsdp_bucket_cap_mb",
        type=int,
        default=0,
        help="FSDP bucket cap in MB (0 uses torch default; ignored if unsupported).",
    )
    p.add_argument(
        "--fsdp-reshard-after-forward",
        "--fsdp_reshard_after_forward",
        type=int,
        choices=[0, 1],
        default=1,
        help="FSDP reshard-after-forward toggle (1=FULL_SHARD, 0=SHARD_GRAD_OP if supported).",
    )
    p.add_argument(
        "--fsdp-limit-all-gathers",
        "--fsdp_limit_all_gathers",
        type=int,
        choices=[0, 1],
        default=1,
        help="FSDP limit_all_gathers toggle (ignored if unsupported).",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation on a held-out shard of --dataset and exit.",
    )
    p.add_argument(
        "--eval-steps",
        type=int,
        default=0,
        help="Number of eval batches to run on the held-out shard (0 disables eval).",
    )
    p.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction of dataset tokens used for eval (taken from the end) when eval is enabled.",
    )
    p.add_argument(
        "--eval-every",
        "--eval_every",
        type=int,
        default=0,
        help=(
            "Run eval every N optimizer steps during training (0 disables). "
            "Requires --eval-steps > 0."
        ),
    )
    p.add_argument(
        "--profile-steps",
        "--profile_steps",
        type=int,
        default=0,
        help="Enable torch.profiler for N steps (CUDA only). Writes trace files outside the repo.",
    )
    p.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Directory to write torch.profiler traces (rank0). Defaults to ~/runs/prof/<run>.",
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
        grad_accum_steps=int(args.grad_accum_steps),
        synthetic=bool(args.synthetic),
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        tokenizer_model=args.tokenizer_model,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        save_every=args.save_every,
        save_optimizer=bool(int(args.save_optimizer)),
        max_checkpoints=int(args.max_checkpoints),
        no_save=bool(args.no_save),
        log_every=int(args.log_every),
        ckpt_dir=args.ckpt_dir,
        resume=args.resume,
        allow_large_cpu=bool(args.allow_large_cpu),
        fsdp_wrap=str(args.fsdp_wrap),
        fsdp_bucket_cap_mb=int(args.fsdp_bucket_cap_mb),
        fsdp_reshard_after_forward=int(args.fsdp_reshard_after_forward),
        fsdp_limit_all_gathers=int(args.fsdp_limit_all_gathers),
        eval_only=bool(args.eval_only),
        eval_steps=int(args.eval_steps),
        eval_split=float(args.eval_split),
        eval_every=int(args.eval_every),
        profile_steps=int(args.profile_steps),
        profile_dir=args.profile_dir,
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
    if a.eval_only and a.dataset is None:
        raise ValueError("--eval-only requires --dataset (and --tokenizer).")
    if a.eval_only and a.eval_steps <= 0:
        raise ValueError("--eval-only requires --eval-steps > 0.")
    if a.eval_steps < 0:
        raise ValueError("--eval-steps must be >= 0.")
    if not (0.0 < float(a.eval_split) < 1.0):
        raise ValueError("--eval-split must be in (0, 1).")
    if int(a.fsdp_bucket_cap_mb) < 0:
        raise ValueError("--fsdp-bucket-cap-mb must be >= 0.")
    if int(a.eval_every) < 0:
        raise ValueError("--eval-every must be >= 0.")
    if int(a.eval_every) > 0 and int(a.eval_steps) <= 0:
        raise ValueError("--eval-every requires --eval-steps > 0.")
    if int(a.grad_accum_steps) < 1:
        raise ValueError("--grad-accum-steps must be >= 1.")
    if a.profile_steps < 0:
        raise ValueError("--profile-steps must be >= 0.")
    if a.profile_steps > 0 and device.type != "cuda":
        raise ValueError("--profile-steps is only supported on CUDA.")

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
    ckpt_dir = Path(a.ckpt_dir)

    # Real data pipeline (non-demo gate): tokenizer + deterministic token stream.
    tokenizer = None
    token_stream = None
    eval_stream = None
    eval_stream_state0 = None
    cfg = base_cfg
    if a.dataset is not None:
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
        elif a.tokenizer in {"spm_load", "spm_model"}:
            if not a.tokenizer_model:
                raise ValueError("--tokenizer-model is required for --tokenizer spm_load.")
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
        if a.eval_only or a.eval_steps > 0:
            n = int(tokens.numel())
            split_at = int(float(n) * (1.0 - float(a.eval_split)))
            split_at = max(0, min(split_at, max(0, n - 1)))
            eval_tokens = tokens[split_at:].contiguous()
            eval_stream = ShardedTokenStream(
                eval_tokens,
                seq_len=a.seq_len,
                batch_size=a.batch_size,
                rank=rank,
                world_size=world,
            )
            # Snapshot initial eval state so periodic eval can reset and remain deterministic
            # across calls (comparable metrics over time).
            eval_stream_state0 = eval_stream.state_dict()

    model = TTTLlamaForCausalLM(cfg)
    if a.precision == "bf16" and device.type == "cuda":
        # Avoid materializing fp32 weights on GPU first (can OOM for very large configs).
        model = model.to(dtype=torch.bfloat16)
    model = model.to(device=device)

    # FSDP wrap (classic FSDP only in v1)
    if a.strategy == "fsdp":
        bucket_cap_mb = int(a.fsdp_bucket_cap_mb)
        fsdp_cfg = FSDPConfig(
            mixed_precision=(a.precision == "bf16"),
            wrap=str(a.fsdp_wrap),
            bucket_cap_mb=(bucket_cap_mb if bucket_cap_mb > 0 else None),
            reshard_after_forward=bool(int(a.fsdp_reshard_after_forward)),
            limit_all_gathers=bool(int(a.fsdp_limit_all_gathers)),
        )
        log_rank0(
            logger,
            "FSDP config: "
            f"wrap={fsdp_cfg.wrap} reshard_after_forward={int(fsdp_cfg.reshard_after_forward)} "
            f"limit_all_gathers={int(bool(fsdp_cfg.limit_all_gathers))} "
            f"bucket_cap_mb={fsdp_cfg.bucket_cap_mb}",
        )
        model = wrap_fsdp(model, cfg=fsdp_cfg)

    optim = AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))

    start_step = 0
    resume_dir = None
    if dist.is_available() and dist.is_initialized():
        # Resolve resume path on rank0 and broadcast to all ranks for consistency.
        obj_list = [""]
        if is_rank0():
            resume_dir = _resolve_resume_dir(resume=a.resume, ckpt_dir=ckpt_dir)
            obj_list[0] = str(resume_dir) if resume_dir is not None else ""
        dist.broadcast_object_list(obj_list, src=0)
        resume_dir = Path(obj_list[0]) if obj_list[0] else None
    else:
        resume_dir = _resolve_resume_dir(resume=a.resume, ckpt_dir=ckpt_dir)

    if resume_dir is not None:
        start_step, meta, rng_state = load_checkpoint(
            resume_dir, model=model, optimizer=optim, scheduler=None
        )
        if rng_state is not None:
            set_rng_state(rng_state)
        if token_stream is not None and (not a.eval_only):
            data_state = (meta.get("extra") or {}).get("data_state", None)
            if isinstance(data_state, dict):
                token_stream.load_state_dict(data_state)
        log_rank0(logger, f"Resumed from {resume_dir} at step={start_step}")

    def _eval_once(*, train_step: int | None) -> tuple[float, float, float, int]:
        if eval_stream is None or eval_stream_state0 is None:
            raise ValueError("--eval requires --dataset (and therefore an eval token stream).")

        # Reset eval stream for deterministic comparisons.
        eval_stream.load_state_dict(eval_stream_state0)

        model.eval()
        autocast_ctx_eval = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (a.precision == "bf16" and device.type == "cuda")
            else nullcontext()
        )
        loss_sum = 0.0
        for i in range(int(a.eval_steps)):
            input_ids = eval_stream.next_batch().to(device=device)
            with torch.no_grad(), autocast_ctx_eval:
                logits, _ = model(input_ids, cache=None, use_dual=True, checkpoint_ttt=False)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, cfg.vocab_size),
                    input_ids[:, 1:].reshape(-1),
                )
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(
                    f"Non-finite eval loss at iter={i} train_step={train_step}: "
                    f"{loss.detach().float().item()}"
                )
            loss_sum += float(loss.detach().float().item())

        # Aggregate across ranks (each rank evaluates the same number of batches).
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor([loss_sum, float(a.eval_steps)], device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            loss_sum = float(t[0].item())
            denom = int(t[1].item())
        else:
            denom = int(a.eval_steps)

        loss_avg = loss_sum / float(max(1, denom))
        bpb = loss_avg / math.log(2.0)
        log10_ppl = loss_avg / math.log(10.0)

        # Restore train mode for the caller.
        model.train()
        return loss_avg, bpb, log10_ppl, denom

    def _append_eval_jsonl(
        *, train_step: int | None, loss: float, bpb: float, log10_ppl: float, denom: int
    ):
        if not is_rank0():
            return
        out_dir = Path(os.path.expanduser("~/runs/eval"))
        out_dir.mkdir(parents=True, exist_ok=True)
        run_name = Path(a.ckpt_dir).name or "run"
        path = out_dir / f"{run_name}_eval.jsonl"
        rec = {
            "time": float(time.time()),
            "train_step": int(train_step) if train_step is not None else None,
            "loss_nats": float(loss),
            "bpb": float(bpb),
            "log10_ppl": float(log10_ppl),
            "eval_steps_total": int(denom),
            "eval_steps_per_rank": int(a.eval_steps),
            "eval_split": float(a.eval_split),
            "dataset": a.dataset,
            "seq_len": int(a.seq_len),
            "batch_size": int(a.batch_size),
            "world_size": int(world),
            "grad_accum_steps": int(a.grad_accum_steps),
            "git_sha": _git_sha(),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    if a.eval_only:
        if resume_dir is None:
            raise ValueError("--eval-only requires a checkpoint (use --resume auto or a path).")

        loss_avg, bpb, log10_ppl, denom = _eval_once(train_step=start_step)
        log_rank0(
            logger,
            f"eval loss={loss_avg:.4f} bpb={bpb:.4f} log10_ppl={log10_ppl:.4f} "
            f"eval_steps={int(denom)} eval_split={a.eval_split:.3f} dataset={a.dataset}",
        )
        _append_eval_jsonl(
            train_step=start_step, loss=loss_avg, bpb=bpb, log10_ppl=log10_ppl, denom=denom
        )

        # Clean shutdown to avoid NCCL/Gloo resource leak warnings.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    model.train()
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    accum = int(a.grad_accum_steps)
    tokens_per_step_global = a.batch_size * a.seq_len * world_size * accum
    if is_rank0() and accum > 1:
        log_rank0(logger, f"Gradient accumulation enabled: grad_accum_steps={accum}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)
    interval_t0 = time.perf_counter()
    interval_steps = 0

    if is_rank0() and a.no_save:
        log_rank0(logger, "Checkpoint saving disabled (--no-save).")

    # Autocast (CUDA bf16) is constant across steps.
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (a.precision == "bf16" and device.type == "cuda")
        else nullcontext()
    )

    # FSDP no_sync (used for gradient accumulation).
    fsdp_no_sync_fn = None
    if a.strategy == "fsdp":
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore

            if isinstance(model, FSDP):
                fsdp_no_sync_fn = model.no_sync
        except Exception:
            fsdp_no_sync_fn = None

    prof = None
    prof_run_dir: Path | None = None
    if a.profile_steps > 0 and device.type == "cuda":
        from torch.profiler import ProfilerActivity, profile

        base = Path(os.path.expanduser("~/runs/prof"))
        run_dir = (
            Path(a.profile_dir)
            if a.profile_dir is not None
            else (base / f"{cfg_name}_T{a.seq_len}_profile_{int(time.time())}")
        )
        if is_rank0():
            run_dir.mkdir(parents=True, exist_ok=True)
            log_rank0(logger, f"Profiler enabled: profile_steps={a.profile_steps} dir={run_dir}")
            prof_run_dir = run_dir
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        prof = profile(
            # CUDA-only to keep traces smaller + faster and avoid multi-rank stalls.
            activities=[ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                # Skip the very first step (init/allocator churn), then profile N steps.
                wait=1, warmup=1, active=int(a.profile_steps), repeat=1
            ),
            record_shapes=False,
            # Memory profiling explodes trace size for large models; keep it off by default.
            profile_memory=False,
            with_stack=False,
        )
        prof.__enter__()

    try:
        for step in range(start_step, a.steps):
            lr_scale = cosine_with_warmup(step, warmup=a.warmup, total=a.steps)
            for pg in optim.param_groups:
                pg["lr"] = a.lr * lr_scale

            optim.zero_grad(set_to_none=True)
            loss_sum = 0.0

            # forward/backward microsteps (accumulate grads); sync only on the final microstep
            for micro in range(accum):
                sync_grads = micro == (accum - 1)
                no_sync_ctx = (
                    fsdp_no_sync_fn()
                    if (fsdp_no_sync_fn is not None and (not sync_grads) and accum > 1)
                    else nullcontext()
                )

                with no_sync_ctx:
                    if a.dataset is not None:
                        assert token_stream is not None
                        input_ids = token_stream.next_batch().to(device=device)
                    else:
                        # synthetic next-token data
                        input_ids = torch.randint(
                            0,
                            cfg.vocab_size,
                            (a.batch_size, a.seq_len),
                            device=device,
                            dtype=torch.long,
                        )

                    with autocast_ctx:
                        logits, _ = model(input_ids, cache=None, use_dual=True, checkpoint_ttt=True)
                        loss = F.cross_entropy(
                            logits[:, :-1, :].reshape(-1, cfg.vocab_size),
                            input_ids[:, 1:].reshape(-1),
                        )

                    if not torch.isfinite(loss.detach()):
                        raise RuntimeError(
                            f"Non-finite loss at step={step} micro={micro}: "
                            f"{loss.detach().float().item()}"
                        )

                    loss_sum += float(loss.detach().float().item())
                    (loss / float(accum)).backward()

            loss_avg = loss_sum / float(max(1, accum))

            grad_norm = None
            if a.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)

            optim.step()
            interval_steps += 1
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
                    f"step={step} loss={loss_avg:.4f} grad_norm={gn:.3e} "
                    f"step_s={step_s:.3f} toks/s={toks_per_s:.1f} "
                    f"peak_mem_gb={peak_mem_gb:.2f} lr={lr:.3e} accum={accum}",
                )

                interval_t0 = time.perf_counter()
                interval_steps = 0
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device=device)

            # Periodic eval (all ranks participate; rank0 logs + writes JSONL).
            if (
                (not a.eval_only)
                and int(a.eval_every) > 0
                and ((step + 1) % int(a.eval_every) == 0)
            ):
                loss_e, bpb_e, log10_ppl_e, denom = _eval_once(train_step=step + 1)
                log_rank0(
                    logger,
                    f"eval@step={step+1} loss={loss_e:.4f} bpb={bpb_e:.4f} "
                    f"log10_ppl={log10_ppl_e:.4f} eval_steps={int(denom)}",
                )
                _append_eval_jsonl(
                    train_step=step + 1,
                    loss=loss_e,
                    bpb=bpb_e,
                    log10_ppl=log10_ppl_e,
                    denom=denom,
                )

            # NOTE: under FSDP, state_dict collection uses collective comms; all ranks must
            # participate even if only rank0 writes metadata.
            if (
                (not a.no_save)
                and a.save_every > 0
                and (step + 1) < a.steps
                and ((step + 1) % a.save_every == 0)
            ):
                extra = {
                    "data_state": token_stream.state_dict() if token_stream is not None else None
                }
                ckpt_path = ckpt_dir / f"step_{step+1:09d}"
                if is_rank0():
                    log_rank0(logger, f"saving checkpoint: step={step+1} dir={ckpt_path}")
                if device.type == "cuda":
                    torch.cuda.synchronize(device=device)
                ckpt_t0 = time.perf_counter()
                save_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=optim if a.save_optimizer else None,
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
                    _write_checkpoint_manifest(
                        ckpt_dir=ckpt_dir,
                        checkpoint_dir=ckpt_path,
                        step=step + 1,
                        cfg_name=cfg_name,
                        tokenizer=a.tokenizer,
                        dataset=a.dataset,
                        world_size=world,
                        seed=int(a.seed),
                        save_optimizer=bool(a.save_optimizer),
                    )
                    _prune_checkpoints(
                        ckpt_dir=ckpt_dir, max_checkpoints=int(a.max_checkpoints), logger=logger
                    )
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

            if prof is not None:
                prof.step()
    finally:
        if prof is not None:
            prof.__exit__(None, None, None)

    if not a.no_save:
        extra = {"data_state": token_stream.state_dict() if token_stream is not None else None}
        ckpt_path = ckpt_dir / f"step_{a.steps:09d}"
        if is_rank0():
            log_rank0(logger, f"saving final checkpoint: step={a.steps} dir={ckpt_path}")
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        ckpt_t0 = time.perf_counter()
        save_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optim if a.save_optimizer else None,
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
            _write_checkpoint_manifest(
                ckpt_dir=ckpt_dir,
                checkpoint_dir=ckpt_path,
                step=a.steps,
                cfg_name=cfg_name,
                tokenizer=a.tokenizer,
                dataset=a.dataset,
                world_size=world,
                seed=int(a.seed),
                save_optimizer=bool(a.save_optimizer),
            )
            _prune_checkpoints(
                ckpt_dir=ckpt_dir, max_checkpoints=int(a.max_checkpoints), logger=logger
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # Clean shutdown to avoid NCCL/Gloo resource leak warnings.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    # NOTE: exporting profiler traces can take a long time (GBs of JSON). Do it only
    # on rank0 *after* tearing down the process group so other ranks aren't stuck in
    # barriers waiting for rank0 (avoids NCCL timeouts).
    if prof is not None and prof_run_dir is not None and is_rank0():
        try:
            # Prefer tensorboard trace handler here: for very large traces,
            # export_chrome_trace() can spend a long time building the full JSON
            # in memory before writing. The TB handler streams more reliably.
            from torch.profiler import tensorboard_trace_handler

            tensorboard_trace_handler(str(prof_run_dir))(prof)
            log_rank0(logger, f"Profiler trace written under: {prof_run_dir}")
        except Exception as e:
            log_rank0(logger, f"Profiler export failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

