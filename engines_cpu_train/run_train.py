#!/usr/bin/env python3
"""CPU quality-first 3DGS training (pure PyTorch, no CUDA)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.trainer import Stage, TrainConfig, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CPU 3DGS trainer - progressive resolution + LPIPS, no CUDA"
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=Path.home() / "tmp" / "blackberry-splats" / "dataset",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "tmp" / "blackberry-splats" / "engines_cpu_train_out",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Short CPU verification (factor 8, few steps)",
    )
    ap.add_argument(
        "--quality",
        action="store_true",
        default=True,
        help="Full progressive schedule (default)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    if args.smoke:
        stages = [
            Stage(data_factor=8, steps=20, max_gaussians_draw=4000, max_radius=16, sh_degree=0),
        ]
    else:
        stages = [
            Stage(data_factor=8, steps=2000, max_gaussians_draw=12000, max_radius=20, sh_degree=0),
            Stage(data_factor=4, steps=8000, max_gaussians_draw=20000, max_radius=24, sh_degree=1),
            Stage(data_factor=2, steps=20000, max_gaussians_draw=30000, max_radius=28, sh_degree=2),
        ]

    cfg = TrainConfig(
        data=str(args.data),
        out=str(args.out),
        stages=stages,
        seed=args.seed,
        device=args.device,
    )
    out = train(cfg)
    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
