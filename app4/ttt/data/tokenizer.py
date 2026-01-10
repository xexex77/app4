from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class SentencePieceTrainConfig:
    vocab_size: int
    model_type: str = "bpe"
    seed: int = 1234


class SentencePieceTokenizer:
    def __init__(self, model_file: str | Path):
        try:
            import sentencepiece as spm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "sentencepiece is required for --tokenizer spm_train/spm_model; "
                "install it via `pip install -e '.[dev]'` (or `pip install sentencepiece`)."
            ) from e

        self._spm = spm.SentencePieceProcessor(model_file=str(model_file))

    @property
    def vocab_size(self) -> int:
        return int(self._spm.get_piece_size())

    def encode(self, text: str) -> list[int]:
        return list(self._spm.encode(text, out_type=int))

    def decode(self, ids: list[int]) -> str:
        return str(self._spm.decode(ids))


def train_sentencepiece(
    *,
    input_path: str | Path,
    model_prefix: str | Path,
    cfg: SentencePieceTrainConfig,
) -> Path:
    """
    Train a SentencePiece model from a text file.

    Returns the created `.model` path (i.e., f"{model_prefix}.model").
    """
    try:
        import sentencepiece as spm  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "sentencepiece is required for --tokenizer spm_train; "
            "install it via `pip install -e '.[dev]'` (or `pip install sentencepiece`)."
        ) from e

    input_path = Path(input_path)
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    args = [
        f"--input={str(input_path)}",
        f"--model_prefix={str(model_prefix)}",
        f"--vocab_size={int(cfg.vocab_size)}",
        f"--model_type={cfg.model_type}",
        "--character_coverage=1.0",
        "--unk_id=0",
        "--bos_id=1",
        "--eos_id=2",
        "--pad_id=3",
        "--hard_vocab_limit=false",
        "--shuffle_input_sentence=false",
    ]
    spm.SentencePieceTrainer.Train(" ".join(args))
    return model_prefix.with_suffix(".model")

