from __future__ import annotations

import torch


def _as_broadcastable(param: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
    if param is None:
        return None
    if param.ndim == 1:
        # (d,) -> (..., d)
        shape = [1] * (x.ndim - 1) + [param.shape[0]]
        return param.view(shape)
    if param.ndim == 2:
        # (H,d) -> (1,H,1,d) for x (B,H,T,d)
        if x.ndim != 4:
            raise ValueError(
                "Expected x to be (B,H,T,d) when param is (H,d); "
                f"got {tuple(x.shape)}"
            )
        h, d = param.shape
        if x.shape[1] != h or x.shape[-1] != d:
            raise ValueError(f"Param (H,d)={param.shape} not compatible with x={tuple(x.shape)}")
        return param.view(1, h, 1, d)
    raise ValueError(f"Unsupported param ndim={param.ndim}; expected 1 or 2")


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """
    LayerNorm over the last dimension.
    Supports x shaped (B,H,T,d) with weight/bias shaped (H,d) or (d,).
    Computes stats in fp32 when x is fp16/bf16.
    """
    w = _as_broadcastable(weight, x)
    b = _as_broadcastable(bias, x)

    x_f = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    mu = x_f.mean(dim=-1, keepdim=True)
    var = x_f.var(dim=-1, keepdim=True, unbiased=False)
    rstd = torch.rsqrt(var + eps)
    xhat = (x_f - mu) * rstd

    y = xhat
    if w is not None:
        y = y * (w.float() if y.dtype == torch.float32 and w.dtype != torch.float32 else w)
    if b is not None:
        y = y + (b.float() if y.dtype == torch.float32 and b.dtype != torch.float32 else b)

    return y.to(dtype=x.dtype)


def layer_norm_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """
    Backward for LayerNorm over the last dim: returns dx given x and upstream dy.
    Supports x shaped (B,H,T,d) with weight shaped (H,d) or (d,).

    Formula (per row):
      xhat = (x - mu) * rstd
      dx = rstd * (dy - mean(dy) - xhat * mean(dy*xhat))
    with dy scaled by weight if affine.
    """
    if x.shape != dy.shape:
        raise ValueError(f"x and dy must match; got x={tuple(x.shape)} dy={tuple(dy.shape)}")

    w = _as_broadcastable(weight, x)

    x_f = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    dy_f = dy.float() if dy.dtype in (torch.float16, torch.bfloat16) else dy

    mu = x_f.mean(dim=-1, keepdim=True)
    var = x_f.var(dim=-1, keepdim=True, unbiased=False)
    rstd = torch.rsqrt(var + eps)
    xhat = (x_f - mu) * rstd

    if w is not None:
        w_f = w.float() if w.dtype in (torch.float16, torch.bfloat16) else w
        dy_f = dy_f * w_f

    d = x.shape[-1]
    dy_sum = dy_f.sum(dim=-1, keepdim=True)
    dy_xhat_sum = (dy_f * xhat).sum(dim=-1, keepdim=True)

    dx = (dy_f - dy_sum / d - xhat * dy_xhat_sum / d) * rstd
    return dx.to(dtype=x.dtype)

