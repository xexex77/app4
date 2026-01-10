import pytest
import torch

from app4.ttt.layers.ttt_linear import ttt_dual_chunk_torch, ttt_primal_chunk_anchored
from app4.ttt.ops.layernorm import layer_norm_backward


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layernorm_backward_matches_autograd(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip("bf16 autograd LN check is run on CUDA only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, t, d = 2, 4, 16
    x = torch.randn(b, t, d, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(d, device=device, dtype=dtype, requires_grad=True)
    b0 = torch.randn(d, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(b, t, d, device=device, dtype=dtype)

    y_ref = torch.nn.functional.layer_norm(x, (d,), w, b0, 1e-5)
    (y_ref * g).sum().backward()
    dx_ref = x.grad.detach().clone()

    x2 = x.detach().clone()
    g2 = g.detach().clone()
    dx = layer_norm_backward(x2, g2, w.detach(), 1e-5)

    assert torch.allclose(dx, dx_ref, atol=2e-3, rtol=2e-3)


def test_dual_matches_anchored_primal_fp32():
    device = torch.device("cpu")
    dtype = torch.float32

    B, H, d, b = 2, 3, 8, 6
    W = torch.randn(B, H, d, d, device=device, dtype=dtype) * 0.02
    K = torch.randn(B, H, b, d, device=device, dtype=dtype)
    V = torch.randn(B, H, b, d, device=device, dtype=dtype)
    Q = torch.randn(B, H, b, d, device=device, dtype=dtype)

    eta = torch.sigmoid(torch.randn(B, H, b, device=device, dtype=dtype)) * 0.1
    ln_weight = torch.ones(H, d, device=device, dtype=dtype)
    ln_bias = torch.zeros(H, d, device=device, dtype=dtype)

    Z_dual, W_dual = ttt_dual_chunk_torch(W, K, V, Q, eta, ln_weight, ln_bias, 1e-5)
    Z_pri, W_pri = ttt_primal_chunk_anchored(W, K, V, Q, eta, ln_weight, ln_bias, 1e-5)

    assert torch.allclose(Z_dual, Z_pri, atol=1e-5, rtol=1e-5)
    assert torch.allclose(W_dual, W_pri, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16 dual/primal test")
def test_dual_matches_anchored_primal_bf16_cuda():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B, H, d, b = 2, 8, 128, 16
    W = (torch.randn(B, H, d, d, device=device, dtype=torch.float32) * 0.02).contiguous()
    K = torch.randn(B, H, b, d, device=device, dtype=dtype)
    V = torch.randn(B, H, b, d, device=device, dtype=dtype)
    Q = torch.randn(B, H, b, d, device=device, dtype=dtype)

    eta = (torch.sigmoid(torch.randn(B, H, b, device=device, dtype=torch.float32)) * 0.1).to(dtype)
    ln_weight = torch.ones(H, d, device=device, dtype=dtype)
    ln_bias = torch.zeros(H, d, device=device, dtype=dtype)

    Z_dual, W_dual = ttt_dual_chunk_torch(W, K, V, Q, eta, ln_weight, ln_bias, 1e-5)
    Z_pri, W_pri = ttt_primal_chunk_anchored(W, K, V, Q, eta, ln_weight, ln_bias, 1e-5)

    # bf16 matmul + LN will introduce small numerical deltas; compare in fp32.
    assert torch.allclose(Z_dual.float(), Z_pri.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(W_dual, W_pri, atol=2e-3, rtol=2e-3)

