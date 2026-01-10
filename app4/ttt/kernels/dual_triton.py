from __future__ import annotations

# Triton kernels are optional. This file provides a hook point for future fusion.
# The correctness-critical reference path is the PyTorch dual form in `layers/ttt_linear.py`.

from dataclasses import dataclass

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@dataclass(frozen=True)
class TritonTTTOptions:
    enabled: bool = False


def triton_available() -> bool:
    return TRITON_AVAILABLE


def ttt_dual_chunk_triton(*args, **kwargs):
    raise NotImplementedError(
        "Triton dual kernel is a stub in this skeleton. Use the PyTorch dual path for correctness. "
        "This hook exists to add fused kernels once dual/primal is validated at bring-up scale."
    )

