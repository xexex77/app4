from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FSDPWrapConfig:
    use_fsdp2_if_available: bool = True
    mixed_precision: bool = True


def maybe_wrap_fsdp(model: nn.Module, *, cfg: FSDPWrapConfig) -> nn.Module:
    """
    Minimal wrapper:
    - tries composable FSDP ("FSDP2") if available
    - falls back to classic FSDP

    This is intentionally small; production runs should tune:
    - auto-wrap policy
    - activation checkpointing
    - CPU offload
    - sharding strategy
    """
    if not torch.distributed.is_available():
        return model

    # --- try composable FSDP (aka FSDP2) ---
    if cfg.use_fsdp2_if_available:
        try:
            from torch.distributed._composable.fsdp import fully_shard  # type: ignore

            fully_shard(model)
            return model
        except Exception:
            pass

    # --- fallback to classic FSDP ---
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy  # type: ignore

        mp = None
        if cfg.mixed_precision and torch.cuda.is_available():
            mp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        return FSDP(model, mixed_precision=mp, sharding_strategy=ShardingStrategy.FULL_SHARD)
    except Exception:
        return model

