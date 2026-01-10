from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FSDPConfig:
    mixed_precision: bool = True


def wrap_fsdp(model: nn.Module, *, cfg: FSDPConfig) -> nn.Module:
    """
    v1 supported distributed sharding path: classic FSDP only.

    Note: this intentionally avoids composable FSDP2 until we implement
    torch.distributed.checkpoint (DCP) end-to-end.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        raise RuntimeError(
            "--strategy fsdp requires torch.distributed to be initialized (use torchrun)."
        )

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Classic FSDP is not available in this torch build.") from e

    # IMPORTANT: tie behavior to the actual module/device, not CUDA availability.
    # Step-7 gate runs `--device cpu` on GPU machines; classic FSDP must not
    # force a CUDA compute device in that case.
    param_device = None
    for p in model.parameters():
        param_device = p.device
        break

    mp = None
    if cfg.mixed_precision and (param_device is not None) and (param_device.type == "cuda"):
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    device_id = None
    if param_device is not None and param_device.type == "cpu":
        device_id = torch.device("cpu")
    elif param_device is not None and param_device.type == "cuda":
        # Avoid relying on global CUDA device; use the local parameter device.
        device_id = param_device

    return FSDP(
        model,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device_id,
    )

