import torch

from app4.ttt.layers.ttt_linear import TTTLinearMixer, TTTLinearMixerConfig


def test_batch_state_isolation_matches_per_sample():
    device = torch.device("cpu")
    dtype = torch.float32

    cfg = TTTLinearMixerConfig(d_model=32, n_heads=4, head_dim=8, b_ttt=4, checkpoint_n_chunks=0)
    mixer = TTTLinearMixer(cfg).to(device=device, dtype=dtype)
    mixer.eval()

    B, T = 2, 8
    x = torch.randn(B, T, cfg.d_model, device=device, dtype=dtype)

    Wb = mixer.init_state(B, device=device, dtype=dtype)
    out_b, Wb_next = mixer(x, Wb, start_pos=0, use_dual=True, checkpoint_ttt=False)

    # run separately
    outs = []
    Ws = []
    for i in range(B):
        Wi = mixer.init_state(1, device=device, dtype=dtype)
        out_i, Wi_next = mixer(x[i : i + 1], Wi, start_pos=0, use_dual=True, checkpoint_ttt=False)
        outs.append(out_i)
        Ws.append(Wi_next)

    out_cat = torch.cat(outs, dim=0)
    W_cat = torch.cat(Ws, dim=0)

    assert torch.allclose(out_b, out_cat, atol=1e-6, rtol=1e-6)
    assert torch.allclose(Wb_next, W_cat, atol=1e-6, rtol=1e-6)

