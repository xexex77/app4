import torch

from app4.ttt.layers.ttt_linear import ttt_dual_chunk_torch


def test_chunk_causality_strict_lower_triangle():
    device = torch.device("cpu")
    dtype = torch.float32

    B, H, d, b = 1, 2, 8, 6
    W = torch.randn(B, H, d, d, device=device, dtype=dtype) * 0.02
    K = torch.randn(B, H, b, d, device=device, dtype=dtype)
    V = torch.randn(B, H, b, d, device=device, dtype=dtype)
    Q = torch.randn(B, H, b, d, device=device, dtype=dtype)

    eta = torch.sigmoid(torch.randn(B, H, b, device=device, dtype=dtype)) * 0.1
    ln_weight = torch.ones(H, d, device=device, dtype=dtype)
    ln_bias = torch.zeros(H, d, device=device, dtype=dtype)

    Z1, _ = ttt_dual_chunk_torch(W, K, V, Q, eta, ln_weight, ln_bias, 1e-5)

    # mutate a FUTURE token; earlier outputs must remain identical
    j = 5
    K2 = K.clone()
    V2 = V.clone()
    Q2 = Q.clone()
    K2[:, :, j, :] += 10.0
    V2[:, :, j, :] -= 7.0
    Q2[:, :, j, :] *= -3.0

    Z2, _ = ttt_dual_chunk_torch(W, K2, V2, Q2, eta, ln_weight, ln_bias, 1e-5)

    # tokens <= 4 must be unchanged
    assert torch.allclose(Z1[:, :, :5, :], Z2[:, :, :5, :], atol=1e-6, rtol=1e-6)

