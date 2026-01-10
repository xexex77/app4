import torch

from app4.ttt.layers.ttt_linear import TTTLinearMixer, TTTLinearMixerConfig
from app4.ttt.model.configs import TTTLMConfig
from app4.ttt.model.llama_ttt import TTTLlamaForCausalLM


def test_fast_weights_state_is_fp32():
    device = torch.device("cpu")

    cfg = TTTLinearMixerConfig(d_model=32, n_heads=4, head_dim=8, b_ttt=4, checkpoint_n_chunks=0)
    mixer = TTTLinearMixer(cfg)
    W = mixer.init_state(2, device=device, dtype=torch.bfloat16)
    assert W.dtype == torch.float32

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
    model = TTTLlamaForCausalLM(mcfg).to(device=device, dtype=torch.float32)
    cache = model.init_cache(batch_size=2, device=device, dtype=torch.bfloat16)
    assert all(w.dtype == torch.float32 for w in cache.W)

    # Optional CUDA check: model in bf16 still keeps fp32 fast weights.
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        model = TTTLlamaForCausalLM(mcfg).to(device=dev, dtype=torch.bfloat16).eval()
        input_ids = torch.randint(0, mcfg.vocab_size, (2, 8), device=dev, dtype=torch.long)
        _, cache2 = model(input_ids, cache=None, use_dual=True, checkpoint_ttt=False)
        assert cache2 is not None
        assert all(w.dtype == torch.float32 for w in cache2.W)

