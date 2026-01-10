from __future__ import annotations

from pathlib import Path

import torch

from app4.ttt.data.tokenizer import Tokenizer


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def tokenize_text(tokenizer: Tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        raise ValueError("Tokenized corpus is too small (<2 tokens).")
    return torch.tensor(ids, dtype=torch.long)


def load_token_ids(path: str | Path, *, tokenizer: Tokenizer) -> torch.Tensor:
    return tokenize_text(tokenizer, load_text(path))

