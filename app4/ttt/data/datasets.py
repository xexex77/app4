from __future__ import annotations

import re
from pathlib import Path

import torch

from app4.ttt.data.tokenizer import Tokenizer


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


_DOC_SPLIT_RE = re.compile(r"\n\s*\n", flags=re.MULTILINE)


def split_documents(text: str) -> list[str]:
    """
    Split a text file into "documents" for packing.

    We use blank-line separation as a simple, format-agnostic heuristic.
    """
    docs = [d.strip() for d in _DOC_SPLIT_RE.split(text) if d.strip()]
    return docs if docs else [text]


def tokenize_packed(tokenizer: Tokenizer, docs: list[str]) -> torch.Tensor:
    """
    Tokenize multiple documents and pack them densely with an EOS separator.
    """
    eos = int(tokenizer.eos_id)
    ids: list[int] = []
    for doc in docs:
        if not doc:
            continue
        ids.extend(tokenizer.encode(doc))
        ids.append(eos)
    if len(ids) < 2:
        raise ValueError("Tokenized corpus is too small (<2 tokens).")
    return torch.tensor(ids, dtype=torch.long)


def load_token_ids(path: str | Path, *, tokenizer: Tokenizer) -> torch.Tensor:
    # Pack multiple "docs" with EOS so sequences are dense and we don't waste tokens
    # on padding between samples.
    docs = split_documents(load_text(path))
    return tokenize_packed(tokenizer, docs)

