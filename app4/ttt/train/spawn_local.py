from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp


def _worker(local_rank: int, world_size: int, init_file: str, argv: list[str]):
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["APP4_DIST_INIT_METHOD"] = f"file:///{init_file.replace(os.sep, '/')}"

    # Run the training module with the provided argv.
    import runpy
    import sys

    sys.argv = ["app4.ttt.train.train", *argv]
    runpy.run_module("app4.ttt.train.train", run_name="__main__")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nproc", type=int, default=2)
    p.add_argument("train_args", nargs=argparse.REMAINDER)
    a = p.parse_args()

    if a.nproc <= 1:
        raise SystemExit("--nproc must be >= 2")

    # Create a file-store init file for rendezvous (works on Windows without TCPStore/libuv).
    tmp = Path(tempfile.gettempdir())
    init_file = str((tmp / f"app4_init_{os.getpid()}.txt").resolve())
    Path(init_file).write_text("app4")

    argv = a.train_args
    if argv and argv[0] == "--":
        argv = argv[1:]

    mp.spawn(
        _worker,
        args=(a.nproc, init_file, argv),
        nprocs=a.nproc,
        join=True,
    )


if __name__ == "__main__":
    main()

