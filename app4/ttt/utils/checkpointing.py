from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
    from torch.distributed.fsdp import StateDictType  # type: ignore
    from torch.distributed.fsdp.api import (  # type: ignore
        ShardedOptimStateDictConfig,
        ShardedStateDictConfig,
    )

    FSDP_AVAILABLE = True
except Exception:
    FSDP_AVAILABLE = False


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_dist() -> bool:
    return _world() > 1


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    step: int,
    cfg: Any | None = None,
    extra: dict[str, Any] | None = None,
    rng_state: Any | None = None,
):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rank = _rank()

    meta = {
        "step": step,
        "world_size": _world(),
        "cfg": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else cfg,
        "extra": extra or {},
    }

    if _is_dist() and FSDP_AVAILABLE and isinstance(model, FSDP):
        # Save sharded model + optimizer state per rank
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            msd = model.state_dict()
        torch.save(msd, path / f"model_rank{rank}.pt")

        if optimizer is not None:
            with FSDP.state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
                optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
            ):
                osd = FSDP.optim_state_dict(model, optimizer)
            torch.save(osd, path / f"optim_rank{rank}.pt")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    else:
        # Single-process (or non-FSDP) fallback
        if rank == 0:
            torch.save(model.state_dict(), path / "model.pt")
            if optimizer is not None:
                torch.save(optimizer.state_dict(), path / "optim.pt")

    if rank == 0:
        if scheduler is not None:
            torch.save(scheduler.state_dict(), path / "sched.pt")
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    # RNG state is not JSON serializable; store it separately (per-rank if distributed).
    if rng_state is not None:
        rng_file = path / (f"rng_rank{rank}.pt" if _is_dist() else "rng.pt")
        torch.save(rng_state, rng_file)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
) -> tuple[int, dict[str, Any], Any | None]:
    path = Path(path)
    rank = _rank()

    def _load_any(p: Path):
        # PyTorch >=2.6 defaults torch.load(weights_only=True), which cannot load
        # objects like ShardedTensor used by FSDP sharded state_dicts. These files
        # are written by this training script, so we opt into weights_only=False.
        try:
            return torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover
            return torch.load(p, map_location="cpu")

    if _is_dist() and FSDP_AVAILABLE and isinstance(model, FSDP):
        msd = _load_any(path / f"model_rank{rank}.pt")
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            model.load_state_dict(msd)

        if optimizer is not None:
            osd = _load_any(path / f"optim_rank{rank}.pt")
            with FSDP.state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
                optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
            ):
                to_load = FSDP.optim_state_dict_to_load(model, optimizer, osd)
            optimizer.load_state_dict(to_load)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    else:
        if rank == 0:
            model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
            if optimizer is not None and (path / "optim.pt").exists():
                optimizer.load_state_dict(torch.load(path / "optim.pt", map_location="cpu"))

    if rank == 0 and scheduler is not None and (path / "sched.pt").exists():
        scheduler.load_state_dict(torch.load(path / "sched.pt", map_location="cpu"))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    step = 0
    meta: dict[str, Any] = {}
    meta_file = path / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        step = int(meta.get("step", 0))

    rng_state = None
    rng_file = path / (f"rng_rank{rank}.pt" if _is_dist() else "rng.pt")
    if rng_file.exists():
        # PyTorch >=2.6 defaults torch.load(weights_only=True), which cannot load
        # arbitrary Python objects (e.g. python/numpy RNG state). These checkpoints
        # are written by this training script, so we opt into weights_only=False.
        try:
            rng_state = torch.load(rng_file, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover
            rng_state = torch.load(rng_file, map_location="cpu")

    return step, meta, rng_state

