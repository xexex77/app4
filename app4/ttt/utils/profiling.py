from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def maybe_profiler(enabled: bool, *, out_dir: str = "runs/prof"):
    if not enabled:
        yield None
        return

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        yield prof

