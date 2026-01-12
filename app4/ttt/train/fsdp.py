from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn


@dataclass
class FSDPConfig:
    mixed_precision: bool = True
    wrap: str = "auto"  # auto | block | block2
    bucket_cap_mb: int | None = None
    reshard_after_forward: bool = True
    limit_all_gathers: bool | None = None


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
        from torch.distributed.fsdp.wrap import (  # type: ignore
            ModuleWrapPolicy,
            transformer_auto_wrap_policy,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Classic FSDP is not available in this torch build.") from e

    # Wrap at the transformer block level to avoid all-gathering the entire model at once.
    # This is critical for very large configs (e.g., ~47B) on 8 GPUs.
    from app4.ttt.model.llama_ttt import TTTLlamaBlock

    wrap_mode = str(cfg.wrap).lower().strip()
    if wrap_mode not in {"auto", "block", "block2"}:
        raise ValueError(f"Unsupported --fsdp-wrap: {cfg.wrap!r}")
    if wrap_mode == "auto":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TTTLlamaBlock},
        )
    elif wrap_mode == "block":
        # Wrap one full transformer block per FSDP unit.
        auto_wrap_policy = ModuleWrapPolicy({TTTLlamaBlock})
    else:
        # block2: coarser wrapping. We keep the model structure unchanged and rely on the
        # outer FSDP wrapper only (no inner auto-wrapping). This reduces the number of
        # NCCL collectives but may increase peak memory (larger all-gathers).
        auto_wrap_policy = None
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print("INFO: --fsdp-wrap=block2 uses single-module wrapping (no inner auto-wrap).")

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

    # Map "reshard_after_forward" onto a supported sharding strategy.
    # - FULL_SHARD: reshard after forward (ZeRO-3 style)
    # - SHARD_GRAD_OP: do not reshard params after forward (closer to ZeRO-2; fewer all-gathers)
    sharding = ShardingStrategy.FULL_SHARD
    if not bool(cfg.reshard_after_forward):
        if hasattr(ShardingStrategy, "SHARD_GRAD_OP"):
            sharding = ShardingStrategy.SHARD_GRAD_OP
        elif torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print("WARN: SHARD_GRAD_OP not supported in this torch build; using FULL_SHARD.")

    # Pass only supported kwargs (torch version dependent).
    import inspect

    sig = inspect.signature(FSDP)
    kwargs = {
        "mixed_precision": mp,
        "sharding_strategy": sharding,
        "auto_wrap_policy": auto_wrap_policy,
        "device_id": device_id,
    }
    if (cfg.limit_all_gathers is not None) and ("limit_all_gathers" in sig.parameters):
        kwargs["limit_all_gathers"] = bool(cfg.limit_all_gathers)
    elif (cfg.limit_all_gathers is not None) and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print("WARN: --fsdp-limit-all-gathers ignored (unsupported by this torch build).")

    if (cfg.bucket_cap_mb is not None) and ("bucket_cap_mb" in sig.parameters):
        kwargs["bucket_cap_mb"] = float(cfg.bucket_cap_mb)
    elif (cfg.bucket_cap_mb is not None) and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print("WARN: --fsdp-bucket-cap-mb ignored (unsupported by this torch build).")

    return FSDP(model, **kwargs)

