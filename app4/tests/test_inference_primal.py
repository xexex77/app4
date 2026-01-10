import torch

from app4.ttt.layers.ttt_linear import ttt_dual_chunk_torch, ttt_primal_step


def test_b1_dual_matches_primal_step():
    device = torch.device("cpu")
    dtype = torch.float32

    B, H, d, b = 2, 3, 8, 1
    W = torch.randn(B, H, d, d, device=device, dtype=dtype) * 0.02
    K = torch.randn(B, H, b, d, device=device, dtype=dtype)
    V = torch.randn(B, H, b, d, device=device, dtype=dtype)
    Q = torch.randn(B, H, b, d, device=device, dtype=dtype)

    eta = torch.sigmoid(torch.randn(B, H, b, device=device, dtype=dtype)) * 0.1
    ln_weight = torch.ones(H, d, device=device, dtype=dtype)
    ln_bias = torch.zeros(H, d, device=device, dtype=dtype)

    Z_dual, W_dual = ttt_dual_chunk_torch(W, K, V, Q, eta, ln_weight, ln_bias, 1e-5)

    z_step, W_step = ttt_primal_step(
        W,
        K[:, :, 0, :],
        V[:, :, 0, :],
        Q[:, :, 0, :],
        eta[:, :, 0],
        ln_weight,
        ln_bias,
        1e-5,
    )

    assert torch.allclose(Z_dual[:, :, 0, :], z_step, atol=1e-6, rtol=1e-6)
    assert torch.allclose(W_dual, W_step, atol=1e-6, rtol=1e-6)

