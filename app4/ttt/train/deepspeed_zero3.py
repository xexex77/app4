from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class DeepSpeedConfig:
    train_batch_size: int
    micro_batch_size_per_gpu: int
    gradient_accumulation_steps: int
    bf16: bool = True
    zero_stage: int = 3


def make_ds_config(cfg: DeepSpeedConfig) -> dict[str, Any]:
    return {
        "train_batch_size": cfg.train_batch_size,
        "train_micro_batch_size_per_gpu": cfg.micro_batch_size_per_gpu,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "bf16": {"enabled": bool(cfg.bf16)},
        "zero_optimization": {
            "stage": int(cfg.zero_stage),
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
    }


def init_deepspeed(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ds_cfg: dict[str, Any],
):
    import deepspeed  # optional dependency

    engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_cfg)
    return engine, optimizer

