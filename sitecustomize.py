"""
Windows dev workaround for torchrun rendezvous on some PyTorch wheels.

Problem:
  torchrun's rendezvous backends (c10d/static) instantiate TCPStore() without
  specifying use_libuv. In some builds, TCPStore defaults to use_libuv=True
  but the wheel is built *without* libuv support, causing torchrun to fail
  before workers launch:
    DistStoreError: use_libuv was requested but PyTorch was built without libuv support

Solution (Windows-only):
  Monkey-patch the TCPStore symbol used by torchrun rendezvous code to default
  use_libuv=False. This is only intended for local dev on Windows; production
  (Linux clusters) are unaffected.
"""

from __future__ import annotations

import os
import sys


def _is_torchrun_process() -> bool:
    argv0 = os.path.basename(sys.argv[0]).lower()
    if "torchrun" in argv0:
        return True
    # Some entrypoints invoke torchrun via `python -m torch.distributed.run`
    return any("torch.distributed.run" in a for a in sys.argv)


def _should_patch() -> bool:
    if sys.platform != "win32":
        return False
    if os.environ.get("APP4_DISABLE_TORCHRUN_TCPSTORE_PATCH", "0") == "1":
        return False
    # Patch for the torchrun parent process, and also for worker processes where
    # distributed env vars are set (env:// rendezvous uses TCPStore too).
    world = os.environ.get("WORLD_SIZE")
    if world is not None:
        try:
            if int(world) > 1:
                return True
        except ValueError:
            pass
    return _is_torchrun_process()


if _should_patch():  # pragma: no cover
    try:
        import torch
        import torch.distributed as dist

        _OrigTCPStore = dist.TCPStore

        class _TCPStoreNoLibuv(_OrigTCPStore):  # type: ignore[misc]
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("use_libuv", False)
                super().__init__(*args, **kwargs)

        # Patch the public dist.TCPStore used by dynamic_rendezvous (standalone).
        dist.TCPStore = _TCPStoreNoLibuv  # type: ignore[assignment]
        # Patch the TCPStore symbol used by env:// rendezvous (module-level import).
        import torch.distributed.rendezvous as _rdzv

        _rdzv.TCPStore = _TCPStoreNoLibuv  # type: ignore[attr-defined]

        import torch.distributed.elastic.rendezvous.c10d_rendezvous_backend as _c10d_rdzv
        import torch.distributed.elastic.rendezvous.static_tcp_rendezvous as _static_rdzv

        _c10d_rdzv.TCPStore = _TCPStoreNoLibuv  # type: ignore[attr-defined]
        _static_rdzv.TCPStore = _TCPStoreNoLibuv  # type: ignore[attr-defined]
    except Exception:
        # Best-effort only; if this fails, torchrun may still be unusable on this env.
        pass

