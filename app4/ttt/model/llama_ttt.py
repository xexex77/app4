from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from app4.ttt.layers.ttt_linear import TTTLinearMixer, TTTLinearMixerConfig
from app4.ttt.model.configs import TTTLMConfig


def _unwrap_fsdp(m: nn.Module) -> nn.Module:
    """
    If `m` is an FSDP wrapper, return the wrapped module; else return `m`.

    This keeps cache init/reset code robust when we use classic FSDP auto-wrapping
    (e.g., wrapping each transformer block) where `self.layers[i]` becomes an FSDP module.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
    except Exception:  # pragma: no cover
        return m
    return m.module if isinstance(m, FSDP) else m


def _is_fsdp(m: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
    except Exception:  # pragma: no cover
        return False
    return isinstance(m, FSDP)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        return x_norm.to(dtype=x.dtype) * self.weight


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TTTLlamaBlock(nn.Module):
    def __init__(self, cfg: TTTLMConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rms_eps)

        mixer_cfg = TTTLinearMixerConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            b_ttt=cfg.b_ttt,
            eta_base=cfg.eta_base,
            lr_gating=cfg.lr_gating,
            lr_gating_per_head=cfg.lr_gating_per_head,
            ln_eps=cfg.ln_eps,
            rope_theta=cfg.rope_theta,
            checkpoint_n_chunks=cfg.checkpoint_n_chunks,
        )
        self.mixer = TTTLinearMixer(mixer_cfg)
        self.mlp = SwiGLUMLP(cfg.d_model, cfg.ffn_dim)

    def init_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return self.mixer.init_state(batch_size, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,  # (B,T,d)
        W: torch.Tensor | None,  # (B,H,hd,hd)
        *,
        start_pos: int,
        use_dual: bool,
        checkpoint_ttt: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if W is None:
            # Lazy cache init (FSDP-friendly): when blocks are FSDP-wrapped, their
            # parameters are only materialized inside the FSDP forward. Initializing
            # cache here avoids accessing sharded/empty parameters outside of that scope.
            W = self.init_state(x.shape[0], device=x.device, dtype=torch.float32)

        h, W = self.mixer(
            self.norm1(x),
            W,
            start_pos=start_pos,
            use_dual=use_dual,
            checkpoint_ttt=checkpoint_ttt,
        )
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, W


@dataclass
class TTTCache:
    W: list[torch.Tensor | None]
    pos: int = 0


class TTTLlamaForCausalLM(nn.Module):
    def __init__(self, cfg: TTTLMConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TTTLlamaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

    def init_cache(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> TTTCache:
        # Fast weights are always stored in fp32 for stability.
        # (Compute casts happen inside the mixer.)
        if any(_is_fsdp(layer) for layer in self.layers):
            # When blocks are FSDP-wrapped, parameters may be sharded/empty outside
            # of the FSDP forward. Defer init to each block's forward.
            W = [None for _ in self.layers]
        else:
            W = [
                layer.init_state(batch_size, device=device, dtype=torch.float32)
                for layer in self.layers
            ]
        return TTTCache(W=W, pos=0)

    def reset_cache(self, cache: TTTCache):
        cache.pos = 0
        if any(_is_fsdp(layer) for layer in self.layers):
            # For FSDP-wrapped blocks, re-init lazily on the next forward.
            cache.W = [None for _ in cache.W]
            return

        for i in range(len(cache.W)):
            assert cache.W[i] is not None
            self.layers[i].mixer.reset_state_(cache.W[i])

    def forward(
        self,
        input_ids: torch.Tensor,  # (B,T)
        *,
        cache: TTTCache | None = None,
        use_dual: bool = True,
        checkpoint_ttt: bool = True,
    ) -> tuple[torch.Tensor, TTTCache | None]:
        b, t = input_ids.shape
        x = self.tok_embeddings(input_ids)

        if cache is None:
            cache = self.init_cache(b, device=x.device, dtype=x.dtype)

        start_pos = cache.pos

        for i, layer in enumerate(self.layers):
            x, cache.W[i] = layer(
                x,
                cache.W[i],
                start_pos=start_pos,
                use_dual=use_dual,
                checkpoint_ttt=checkpoint_ttt,
            )

        x = self.norm(x)
        logits = self.lm_head(x)

        cache.pos = start_pos + t
        return logits, cache

