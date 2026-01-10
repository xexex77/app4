import torch

from app4.ttt.layers.ttt_linear import TTTLinearMixer, TTTLinearMixerConfig
from app4.ttt.model.configs import TTTLMConfig
from app4.ttt.model.llama_ttt import TTTLlamaForCausalLM


def test_w0_used_for_init_and_reset_and_no_shared_storage():
    device = torch.device("cpu")
    dtype = torch.float32

    # Mixer-level init_state should equal W0 and not share storage across batch.
    cfg = TTTLinearMixerConfig(d_model=32, n_heads=4, head_dim=8, b_ttt=4, checkpoint_n_chunks=0)
    mixer = TTTLinearMixer(cfg).to(device=device, dtype=dtype)

    W = mixer.init_state(2, device=device, dtype=dtype)
    assert torch.allclose(W[0], mixer.W0, atol=0.0, rtol=0.0)
    assert torch.allclose(W[1], mixer.W0, atol=0.0, rtol=0.0)

    before = W.clone()
    W[0, 0, 0, 0] += 1.0
    # If batch storage was shared, W[1,0,0,0] would change too.
    assert torch.allclose(W[1, 0, 0, 0], before[1, 0, 0, 0], atol=0.0, rtol=0.0)

    # Model-level reset_cache must reset to learned prior (W0), not zeros.
    mcfg = TTTLMConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=4,
        head_dim=8,
        ffn_dim=64,
        max_seq_len=64,
        tie_embeddings=False,
        checkpoint_n_chunks=0,
    )
    model = TTTLlamaForCausalLM(mcfg).to(device=device, dtype=dtype)
    cache = model.init_cache(batch_size=2, device=device, dtype=dtype)

    # Corrupt state then reset and verify equality to each layer's W0.
    cache.W[0].add_(0.123)
    cache.W[1].mul_(1.1)
    model.reset_cache(cache)

    for li, layer in enumerate(model.layers):
        w0 = layer.mixer.W0.to(dtype=dtype)
        assert torch.allclose(cache.W[li][0], w0, atol=0.0, rtol=0.0)
        assert torch.allclose(cache.W[li][1], w0, atol=0.0, rtol=0.0)

