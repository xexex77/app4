from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from app4.ttt.ops.layernorm import layer_norm, layer_norm_backward
from app4.ttt.ops.rope import RotaryEmbedding, apply_rope


@dataclass
class TTTLinearMixerConfig:
    d_model: int
    n_heads: int
    head_dim: int
    b_ttt: int = 16
    eta_base: float = 1e-2
    lr_gating: bool = True
    lr_gating_per_head: bool = False
    ln_eps: float = 1e-5
    rope_theta: float = 10000.0
    checkpoint_n_chunks: int = 8  # 0 disables through-time checkpointing


def _outer(u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # u,k: (B,H,d) -> (B,H,d,d) = u^T k
    return u.unsqueeze(-1) * k.unsqueeze(-2)


def ttt_dual_chunk_torch(
    W: torch.Tensor,  # (B,H,d,d)
    K: torch.Tensor,  # (B,H,b,d)
    V: torch.Tensor,  # (B,H,b,d)
    Q: torch.Tensor,  # (B,H,b,d)
    eta: torch.Tensor,  # (B,H,b)
    ln_weight: torch.Tensor,  # (H,d)
    ln_bias: torch.Tensor,  # (H,d)
    ln_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dual form within one chunk (size b), per head, row-vector convention:
      f(z;W) = z + LN(z @ W^T)   (equivalent to z + LN(W z) in column convention)

    Update gradients anchored at W_mod (W at chunk start):
      tK = K @ W^T
      u  = LN_backward(tK, 2*(K + LN(tK) - V))

    Causal output uses strict lower triangle (token i sees updates from tokens < i):
      S = Q @ K^T
      corr = tril(S, -1) @ (eta * u)
      TQ = Q @ W^T - corr
      Z = Q + LN(TQ)

    Fast-weight update after chunk:
      W_next = W - (eta*u)^T @ K

    Semantics invariant:
    - u is computed at the chunk-start anchor W_mod (W input to this function)
    - outputs are causally updated (token i sees updates from tokens < i via tril(S, -1))
    """
    b = K.shape[2]
    if b == 0:
        return Q, W

    # Compute matmuls in activation dtype for speed, but keep state/update in fp32.
    Wc = W.to(dtype=K.dtype)

    # --- compute u for each token using ANCHOR weights W ---
    tK = torch.matmul(K, Wc.transpose(-1, -2))  # (B,H,b,d)
    nK = layer_norm(tK, ln_weight, ln_bias, ln_eps)
    y = K + nK
    g = (2.0 / float(b)) * (y - V)
    u = layer_norm_backward(tK, g, ln_weight, ln_eps)  # (B,H,b,d)

    # --- causal application to Q path ---
    S = torch.matmul(Q, K.transpose(-1, -2))  # (B,H,b,b), S[i,j] = q_i · k_j
    S_tril = torch.tril(S, diagonal=-1)

    U_eta = u * eta.unsqueeze(-1)  # (B,H,b,d)
    corr = torch.matmul(S_tril, U_eta)  # (B,H,b,d)

    TQ = torch.matmul(Q, Wc.transpose(-1, -2)) - corr
    nQ = layer_norm(TQ, ln_weight, ln_bias, ln_eps)
    Z = Q + nQ

    # --- W update ---
    dW = torch.matmul(U_eta.float().transpose(-2, -1), K.float())  # (B,H,d,d) fp32
    W_next = W - dW
    return Z, W_next


def ttt_primal_chunk_anchored(
    W: torch.Tensor,  # (B,H,d,d)
    K: torch.Tensor,  # (B,H,b,d)
    V: torch.Tensor,  # (B,H,b,d)
    Q: torch.Tensor,  # (B,H,b,d)
    eta: torch.Tensor,  # (B,H,b)
    ln_weight: torch.Tensor,  # (H,d)
    ln_bias: torch.Tensor,  # (H,d)
    ln_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Primal sequential reference that matches the dual form exactly:
    - u_i computed w.r.t ANCHOR W (W at chunk start)
    - outputs computed with causally-updated W_i (updates from tokens < i)
    - update applied AFTER producing output token i
    """
    b = K.shape[2]
    if b == 0:
        return Q, W

    Wc = W.to(dtype=K.dtype)

    # u computed once using anchor W
    tK = torch.matmul(K, Wc.transpose(-1, -2))
    nK = layer_norm(tK, ln_weight, ln_bias, ln_eps)
    y = K + nK
    g = (2.0 / float(b)) * (y - V)
    u = layer_norm_backward(tK, g, ln_weight, ln_eps)  # (B,H,b,d)

    W_i = W
    z_out = []
    for i in range(b):
        q_i = Q[:, :, i, :]  # (B,H,d)
        Wi_c = W_i.to(dtype=q_i.dtype)
        tQ = torch.matmul(q_i.unsqueeze(-2), Wi_c.transpose(-1, -2)).squeeze(-2)  # (B,H,d)
        nQ = layer_norm(tQ.unsqueeze(2), ln_weight, ln_bias, ln_eps).squeeze(2)  # (B,H,d)
        z_i = q_i + nQ
        z_out.append(z_i)

        # update AFTER output (strictly causal)
        u_i = u[:, :, i, :]
        k_i = K[:, :, i, :]
        eta_i = eta[:, :, i].unsqueeze(-1).unsqueeze(-1).float()  # (B,H,1,1)
        dW_i = eta_i * _outer(u_i.float(), k_i.float())
        W_i = W_i - dW_i

    Z = torch.stack(z_out, dim=2)  # (B,H,b,d)
    return Z, W_i


def ttt_primal_step(
    W: torch.Tensor,  # (B,H,d,d)
    k: torch.Tensor,  # (B,H,d)
    v: torch.Tensor,  # (B,H,d)
    q: torch.Tensor,  # (B,H,d)
    eta: torch.Tensor,  # (B,H)
    ln_weight: torch.Tensor,  # (H,d)
    ln_bias: torch.Tensor,  # (H,d)
    ln_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode step (b=1): output uses current W; then apply the update for this token.
    Anchor == current W when b=1.
    """
    Wc = W.to(dtype=q.dtype)

    # output path first (strict causality)
    tQ = torch.matmul(q.unsqueeze(-2), Wc.transpose(-1, -2)).squeeze(-2)  # (B,H,d)
    nQ = layer_norm(tQ.unsqueeze(2), ln_weight, ln_bias, ln_eps).squeeze(2)
    z = q + nQ

    # update path
    tK = torch.matmul(k.unsqueeze(-2), Wc.transpose(-1, -2)).squeeze(-2)
    nK = layer_norm(tK.unsqueeze(2), ln_weight, ln_bias, ln_eps).squeeze(2)
    y = k + nK
    g = 2.0 * (y - v)
    u = layer_norm_backward(tK.unsqueeze(2), g.unsqueeze(2), ln_weight, ln_eps).squeeze(2)

    dW = eta.float().unsqueeze(-1).unsqueeze(-1) * _outer(u.float(), k.float())
    W_next = W - dW
    return z, W_next


class TTTLinearMixer(nn.Module):
    def __init__(self, cfg: TTTLinearMixerConfig):
        super().__init__()
        if cfg.n_heads * cfg.head_dim != cfg.d_model:
            raise ValueError("n_heads * head_dim must equal d_model")
        if cfg.b_ttt <= 0:
            raise ValueError("b_ttt must be > 0")

        self.cfg = cfg

        self.w_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.w_k = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.w_v = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.w_o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        # Learned fast-weight initialization (per head): W0 ∈ R^{H×d×d}
        # This is the learned prior to reset to (not zero-init).
        self.W0 = nn.Parameter(torch.empty(cfg.n_heads, cfg.head_dim, cfg.head_dim))
        nn.init.normal_(self.W0, mean=0.0, std=0.02)

        # LN params for f(z;W) = z + LN(Wz): per head, per dim
        self.ln_weight = nn.Parameter(torch.ones(cfg.n_heads, cfg.head_dim))
        self.ln_bias = nn.Parameter(torch.zeros(cfg.n_heads, cfg.head_dim))

        # gating: eta(x) = eta_base * sigmoid(theta_lr · x)
        out = cfg.n_heads if cfg.lr_gating_per_head else 1
        self.theta_lr = nn.Linear(cfg.d_model, out, bias=False)

        self.rope = RotaryEmbedding(cfg.head_dim, theta=cfg.rope_theta)

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # IMPORTANT: do not use .expand without materializing a copy; that would share
        # storage across the batch dimension and corrupt per-sequence isolation.
        w0 = self.W0.to(device=device, dtype=torch.float32)
        return w0.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()

    @torch.no_grad()
    def reset_state_(self, W: torch.Tensor):
        """
        In-place reset of a fast-weight state tensor to the learned prior W0.
        W: (B,H,d,d)
        """
        w0 = self.W0.to(device=W.device, dtype=W.dtype)
        W.copy_(w0.unsqueeze(0).expand_as(W))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,H*d) -> (B,H,T,d)
        b, t, _ = x.shape
        x = x.view(b, t, self.cfg.n_heads, self.cfg.head_dim)
        return x.transpose(1, 2).contiguous()

    def _eta(self, x_model: torch.Tensor) -> torch.Tensor:
        # returns (B,H,T)
        b, t, _ = x_model.shape
        if not self.cfg.lr_gating:
            return torch.full(
                (b, self.cfg.n_heads, t),
                float(self.cfg.eta_base),
                device=x_model.device,
                dtype=x_model.dtype,
            )

        logits = self.theta_lr(x_model)  # (B,T,1) or (B,T,H)
        eta = float(self.cfg.eta_base) * torch.sigmoid(logits)

        if eta.shape[-1] == 1:
            eta = eta.expand(b, t, self.cfg.n_heads)
        eta = eta.transpose(1, 2).contiguous()  # (B,H,T)
        return eta.to(dtype=x_model.dtype)

    def _qkv_eta(self, x: torch.Tensor, start_pos: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B,T,d_model) -> q,k,v: (B,H,T,d), eta: (B,H,T)
        q = self._shape(self.w_q(x))
        k = self._shape(self.w_k(x))
        v = self._shape(self.w_v(x))
        eta = self._eta(x)

        t = x.shape[1]
        positions = torch.arange(start_pos, start_pos + t, device=x.device)
        cos, sin = self.rope.cos_sin(positions, dtype=q.dtype, device=q.device)
        q, k = apply_rope(q, k, cos, sin)
        return q, k, v, eta

    def _project_out(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B,H,T,d) -> (B,T,d_model)
        b, h, t, d = z.shape
        y = z.transpose(1, 2).contiguous().view(b, t, h * d)
        return self.w_o(y)

    def forward(
        self,
        x: torch.Tensor,  # (B,T,d_model)
        W: torch.Tensor,  # (B,H,d,d)
        *,
        start_pos: int,
        use_dual: bool,
        checkpoint_ttt: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("x must be (B,T,d_model)")
        if W.ndim != 4:
            raise ValueError("W must be (B,H,d,d)")

        _, tlen, _ = x.shape
        if tlen == 0:
            return x, W

        if self.training and not use_dual:
            raise ValueError(
                "Training must use anchored MBGD semantics (dual form). "
                "Call with use_dual=True for training/prefill; use_dual=False is reserved for decode/online GD."
            )

        if not use_dual:
            # primal sequential (decode)
            q, k, v, eta = self._qkv_eta(x, start_pos=start_pos)
            z_tokens = []
            W_i = W
            for i in range(tlen):
                z_i, W_i = ttt_primal_step(
                    W_i,
                    k[:, :, i, :],
                    v[:, :, i, :],
                    q[:, :, i, :],
                    eta[:, :, i],
                    self.ln_weight,
                    self.ln_bias,
                    self.cfg.ln_eps,
                )
                z_tokens.append(z_i)
            z = torch.stack(z_tokens, dim=2)  # (B,H,T,d)
            return self._project_out(z), W_i

        # dual (prefill / training)
        seg_tokens = (
            self.cfg.b_ttt * self.cfg.checkpoint_n_chunks
            if (self.training and checkpoint_ttt and self.cfg.checkpoint_n_chunks > 0)
            else tlen
        )

        outs = []
        W_i = W

        for seg_start in range(0, tlen, seg_tokens):
            seg_end = min(seg_start + seg_tokens, tlen)
            x_seg = x[:, seg_start:seg_end, :]

            def _segment_fn(W_in: torch.Tensor, x_in: torch.Tensor):
                q, k, v, eta = self._qkv_eta(x_in, start_pos=start_pos + seg_start)

                z_chunks = []
                W_local = W_in
                for cs in range(0, x_in.shape[1], self.cfg.b_ttt):
                    ce = min(cs + self.cfg.b_ttt, x_in.shape[1])
                    z_c, W_local = ttt_dual_chunk_torch(
                        W_local,
                        k[:, :, cs:ce, :],
                        v[:, :, cs:ce, :],
                        q[:, :, cs:ce, :],
                        eta[:, :, cs:ce],
                        self.ln_weight,
                        self.ln_bias,
                        self.cfg.ln_eps,
                    )
                    z_chunks.append(z_c)
                z_seg = torch.cat(z_chunks, dim=2)
                out_seg = self._project_out(z_seg)
                return out_seg, W_local

            if self.training and checkpoint_ttt and self.cfg.checkpoint_n_chunks > 0:
                out_seg, W_i = ckpt.checkpoint(_segment_fn, W_i, x_seg, use_reentrant=False)
            else:
                out_seg, W_i = _segment_fn(W_i, x_seg)

            outs.append(out_seg)

        out = torch.cat(outs, dim=1)  # (B,T,d_model)
        return out, W_i

