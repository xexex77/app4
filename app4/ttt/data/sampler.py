from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


def dist_rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


@dataclass
class ShardedTokenStreamState:
    pos: int


class ShardedTokenStream:
    """
    Minimal deterministic token stream:
    - shard tokens deterministically by (rank, world_size) into contiguous blocks
    - iterate sequentially within the shard
    - resume by restoring `pos`

    This is intentionally simple: correctness + deterministic resume first, then performance.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        *,
        seq_len: int,
        batch_size: int,
        rank: int,
        world_size: int,
    ):
        if tokens.ndim != 1:
            raise ValueError("tokens must be 1D")
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if world_size <= 0 or not (0 <= rank < world_size):
            raise ValueError("invalid rank/world_size")

        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)

        n = int(tokens.numel())
        start = (n * self.rank) // self.world_size
        end = (n * (self.rank + 1)) // self.world_size

        # Keep shard on CPU; the training loop moves batches to the target device.
        full = tokens.detach().cpu().contiguous()
        shard = full[start:end]
        # If the corpus is extremely small relative to world_size, a rank's shard can be empty.
        # Fall back to the full corpus to avoid division-by-zero / shape issues. This keeps
        # deterministic behavior and is fine for bring-up on tiny datasets.
        if shard.numel() == 0:
            shard = full
        self._tokens = shard
        self._pos = 0
        self._base = torch.arange(self.seq_len, dtype=torch.long)

    def state_dict(self) -> dict[str, int]:
        return {"pos": int(self._pos)}

    def load_state_dict(self, state: dict[str, int]):
        self._pos = int(state.get("pos", 0))
        if not (0 <= self._pos < int(self._tokens.numel())):
            raise ValueError("invalid token stream position in checkpoint")

    def next_batch(self) -> torch.Tensor:
        """
        Returns:
          input_ids: (B, T) int64 on CPU
        """
        out = torch.empty((self.batch_size, self.seq_len), dtype=torch.long)
        n = int(self._tokens.numel())

        if n <= 0:
            raise RuntimeError("Token stream is empty.")

        for b in range(self.batch_size):
            # Support tiny shards (n < seq_len) by wrapping around deterministically.
            idx = (self._base + int(self._pos)) % n
            out[b].copy_(self._tokens.index_select(0, idx))
            self._pos = int((int(self._pos) + self.seq_len) % n)

        return out

