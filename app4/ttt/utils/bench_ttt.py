from __future__ import annotations

import argparse
import time

import torch

from app4.ttt.layers.ttt_linear import ttt_dual_chunk_torch, ttt_primal_chunk_anchored


def _device_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    if s == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


@torch.no_grad()
def _bench_one(
    *,
    name: str,
    fn,
    device: torch.device,
    iters: int,
    warmup: int,
    tokens_per_iter: int,
) -> tuple[float, int]:
    for _ in range(warmup):
        fn()
    _device_sync(device)

    max_mem = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _device_sync(device)
    dt = time.perf_counter() - t0

    if device.type == "cuda":
        max_mem = int(torch.cuda.max_memory_allocated(device=device))

    toks_per_s = float(tokens_per_iter * iters) / max(1e-9, dt)
    print(f"{name:>8}  {toks_per_s:12.1f} tokens/s   peak_mem={max_mem/1e6:8.1f} MB")
    return toks_per_s, max_mem


def main():
    p = argparse.ArgumentParser(
        description="Microbenchmark: TTT dual vs primal (chunk) throughput."
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--assert-speedup", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    B, H, d, b = int(args.batch), int(args.heads), int(args.head_dim), int(args.chunk)

    # Fast-weight state is fp32; compute tensors are dtype (bf16/fp16/fp32).
    W = (torch.randn(B, H, d, d, device=device, dtype=torch.float32) * 0.02).contiguous()
    K = torch.randn(B, H, b, d, device=device, dtype=dtype)
    V = torch.randn(B, H, b, d, device=device, dtype=dtype)
    Q = torch.randn(B, H, b, d, device=device, dtype=dtype)

    eta = (torch.sigmoid(torch.randn(B, H, b, device=device, dtype=torch.float32)) * 0.1).to(dtype)
    ln_weight = torch.ones(H, d, device=device, dtype=dtype)
    ln_bias = torch.zeros(H, d, device=device, dtype=dtype)
    ln_eps = 1e-5

    def run_dual():
        return ttt_dual_chunk_torch(W, K, V, Q, eta, ln_weight, ln_bias, ln_eps)

    def run_primal():
        return ttt_primal_chunk_anchored(W, K, V, Q, eta, ln_weight, ln_bias, ln_eps)

    tokens_per_iter = B * b
    print(f"shape: B={B} H={H} d={d} b={b} dtype={dtype} device={device}")
    dual_tps, _ = _bench_one(
        name="dual",
        fn=run_dual,
        device=device,
        iters=int(args.iters),
        warmup=int(args.warmup),
        tokens_per_iter=tokens_per_iter,
    )
    pri_tps, _ = _bench_one(
        name="primal",
        fn=run_primal,
        device=device,
        iters=int(args.iters),
        warmup=int(args.warmup),
        tokens_per_iter=tokens_per_iter,
    )

    speedup = dual_tps / max(1e-9, pri_tps)
    print(f"speedup: {speedup:.2f}x (dual/primal)")

    if args.assert_speedup and speedup < float(args.assert_speedup):
        raise SystemExit(
            f"FAIL: speedup {speedup:.2f}x < required {float(args.assert_speedup):.2f}x "
            f"at B={B},H={H},d={d},b={b}."
        )


if __name__ == "__main__":
    main()

