from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim

    def cos_sin(self, positions: torch.Tensor, *, dtype: torch.dtype, device: torch.device):
        # positions: (T,)
        freqs = torch.einsum("t,f->tf", positions.to(torch.float32), self.inv_freq)  # (T, dim/2)
        cos = torch.cos(freqs).to(dtype=dtype, device=device)
        sin = torch.sin(freqs).to(dtype=dtype, device=device)
        return cos, sin


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: (B,H,T,D)
    cos, sin: (T, D/2)
    """
    _, _, t, d = q.shape
    if d % 2 != 0:
        raise ValueError("RoPE requires even head dim")

    cos = cos.view(1, 1, t, d // 2)
    sin = sin.view(1, 1, t, d // 2)

    def _rope(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

    return _rope(q), _rope(k)

