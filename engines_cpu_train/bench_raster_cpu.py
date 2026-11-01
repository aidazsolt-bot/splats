#!/usr/bin/env python3
"""Compare legacy loop vs batched CPU raster (forward + one backward)."""

from __future__ import annotations

import argparse
import time

import torch

from raster.diff_gauss_cpu import rasterize_gaussians_cpu


def _synth(n: int, w: int, h: int, device: torch.device):
    means = torch.randn(n, 3, device=device) * 2.0
    means[:, 2] = means[:, 2].abs() + 2.0
    scales = torch.full((n, 3), -3.0, device=device)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    opac = torch.zeros(n, device=device)
    dc = torch.randn(n, 3, device=device) * 0.2
    view = torch.eye(4, device=device)
    K = torch.tensor(
        [[0.8 * w, 0.0, w * 0.5], [0.0, 0.8 * h, h * 0.5], [0.0, 0.0, 1.0]],
        device=device,
    )
    return means, scales, quats, opac, dc, view, K


def _run(method: str, n: int, w: int, h: int, reps: int, batch: int) -> float:
    device = torch.device("cpu")
    means, scales, quats, opac, dc, view, K = _synth(n, w, h, device)
    means = means.requires_grad_(True)
    # warmup
    rgb, alph = rasterize_gaussians_cpu(
        means, scales, quats, opac, dc, None, view, K, w, h,
        sh_degree=0, max_gaussians=n, max_radius=16,
        method=method, batch_size=batch, tile_size=0, num_threads=8,
    )
    (rgb.mean() + alph.mean()).backward()
    means.grad = None

    t0 = time.perf_counter()
    for _ in range(reps):
        means_ = means.detach().requires_grad_(True)
        rgb, alph = rasterize_gaussians_cpu(
            means_, scales, quats, opac, dc, None, view, K, w, h,
            sh_degree=0, max_gaussians=n, max_radius=16,
            method=method, batch_size=batch, tile_size=0, num_threads=8,
        )
        (rgb.mean() + alph.mean()).backward()
    return (time.perf_counter() - t0) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--w", type=int, default=160)
    ap.add_argument("--h", type=int, default=240)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    t_loop = _run("loop", args.n, args.w, args.h, args.reps, args.batch)
    t_bat = _run("batched", args.n, args.w, args.h, args.reps, args.batch)
    print(
        f"n={args.n} {args.w}x{args.h}  loop={t_loop:.3f}s  "
        f"batched={t_bat:.3f}s  speedup={t_loop / max(t_bat, 1e-9):.2f}x"
    )


if __name__ == "__main__":
    main()
