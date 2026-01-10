from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ConfigName = Literal["debug_tiny", "bringup_1p3b", "mid_35b", "ttt_llama_47b", "ttt_llama_70b"]


@dataclass
class TTTLMConfig:
    # core
    vocab_size: int = 128_256
    d_model: int = 8192
    n_layers: int = 48
    n_heads: int = 64
    head_dim: int = 128
    ffn_dim: int = 28_672

    # norms / rope
    rms_eps: float = 1e-5
    ln_eps: float = 1e-5
    rope_theta: float = 500_000.0

    # TTT
    b_ttt: int = 16
    eta_base: float = 1e-2
    lr_gating: bool = True
    lr_gating_per_head: bool = False
    checkpoint_n_chunks: int = 8  # 0 disables through-time checkpointing

    # training helpers
    max_seq_len: int = 4096
    tie_embeddings: bool = True

    def validate(self):
        assert self.n_heads * self.head_dim == self.d_model, "n_heads*head_dim must equal d_model"
        assert self.b_ttt > 0
        assert self.n_layers > 0


def get_config(
    name: ConfigName,
    *,
    vocab_size: int | None = None,
    rope_theta: float | None = None,
) -> TTTLMConfig:
    if name == "debug_tiny":
        cfg = TTTLMConfig(
            vocab_size=4096,
            d_model=256,
            n_layers=4,
            n_heads=4,
            head_dim=64,
            ffn_dim=1024,
            rope_theta=10000.0,
            max_seq_len=512,
            b_ttt=16,
            eta_base=1e-2,
            checkpoint_n_chunks=2,
            tie_embeddings=True,
        )
    elif name == "bringup_1p3b":
        cfg = TTTLMConfig(
            vocab_size=128_256,
            d_model=2048,
            n_layers=24,
            n_heads=16,
            head_dim=128,
            ffn_dim=5504,
            rope_theta=10000.0,
            max_seq_len=2048,
            b_ttt=16,
            eta_base=1e-2,
            checkpoint_n_chunks=8,
        )
    elif name == "mid_35b":
        cfg = TTTLMConfig(
            vocab_size=128_256,
            d_model=6144,
            n_layers=48,
            n_heads=48,
            head_dim=128,
            ffn_dim=22016,
            rope_theta=500_000.0,
            max_seq_len=4096,
            b_ttt=16,
            eta_base=1e-2,
            checkpoint_n_chunks=8,
        )
    elif name == "ttt_llama_47b":
        cfg = TTTLMConfig(
            vocab_size=128_256,
            d_model=8192,
            n_layers=48,
            n_heads=64,
            head_dim=128,
            ffn_dim=28672,
            rope_theta=500_000.0,
            max_seq_len=4096,
            b_ttt=16,
            eta_base=1e-2,
            checkpoint_n_chunks=8,
        )
    elif name == "ttt_llama_70b":
        cfg = TTTLMConfig(
            vocab_size=128_256,
            d_model=8192,
            n_layers=80,
            n_heads=64,
            head_dim=128,
            ffn_dim=28672,
            rope_theta=500_000.0,
            max_seq_len=4096,
            b_ttt=16,
            eta_base=1e-2,
            checkpoint_n_chunks=8,
        )
    else:
        raise ValueError(f"Unknown config: {name}")

    if vocab_size is not None:
        cfg.vocab_size = vocab_size
    if rope_theta is not None:
        cfg.rope_theta = rope_theta

    cfg.validate()
    return cfg

