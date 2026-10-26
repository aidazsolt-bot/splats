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
        default=Path("/usr/src/TestAdnroidNDKSDK/blackberry-splats") / "dataset",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "cpu_train_smoke",
        help="Output directory (default: workspace cpu_train_smoke/)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Short CPU verification (factor 8, few steps)",
    )
    ap.add_argument(
        "--hires",
        action="store_true",
        help="Full progressive schedule ending at data_factor=1 (native res)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Continue from ckpt_latest.pt in --out (same settings required)",
    )
    ap.add_argument(
        "--ckpt-every",
        type=int,
        default=50,
        help="Write ckpt_latest.pt every N local steps (0=stage ends / interrupt only)",
    )
    args = ap.parse_args()

    if args.smoke:
        stages = [
            Stage(data_factor=8, steps=20, max_gaussians_draw=4000, max_radius=16, sh_degree=0),
        ]
    elif args.hires:
        stages = [
            Stage(data_factor=8, steps=2000, max_gaussians_draw=12000, max_radius=20, sh_degree=0),
            Stage(data_factor=4, steps=8000, max_gaussians_draw=20000, max_radius=24, sh_degree=1),
            Stage(data_factor=2, steps=15000, max_gaussians_draw=30000, max_radius=28, sh_degree=2),
            Stage(data_factor=1, steps=10000, max_gaussians_draw=40000, max_radius=32, sh_degree=2),
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
        resume=args.resume,
        ckpt_every=args.ckpt_every,
    )
    out = train(cfg)
    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
