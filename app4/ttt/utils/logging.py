from __future__ import annotations

import logging
import os


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_rank0() -> bool:
    return get_rank() == 0


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("app4")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][rank=%(rank)s] %(message)s"))
    logger.addHandler(h)

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = get_rank()
        return record

    logging.setLogRecordFactory(record_factory)
    return logger


def log_rank0(logger: logging.Logger, msg: str):
    if is_rank0():
        logger.info(msg)

