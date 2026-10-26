#!/usr/bin/env python3
"""Train photorealistic 3D Gaussians with gsplat + PyTorch CUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.trainer import TrainConfig, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CUDA/Torch 3DGS trainer (alpha-masked, COLMAP RGBA)"
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=Path.home() / "tmp" / "blackberry-splats" / "dataset",
        help="COLMAP root with cameras.txt / images.txt / points3D.txt / images/",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "tmp" / "blackberry-splats" / "engines_cuda_out",
    )
    ap.add_argument("--data-factor", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--alpha-lambda", type=float, default=0.1)
    ap.add_argument("--no-densify", action="store_true")
    ap.add_argument("--no-random-bkgd", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run (500 steps) to verify the stack on a GPU box",
    )
    args = ap.parse_args()

    save_steps = (7000, 15000, 30000)
    max_steps = args.max_steps
    if args.smoke:
        max_steps = 500
        save_steps = (500,)

    cfg = TrainConfig(
        data=str(args.data),
        out=str(args.out),
        data_factor=args.data_factor,
        max_steps=max_steps,
        save_steps=save_steps,
        test_every=args.test_every,
        sh_degree=args.sh_degree,
        ssim_lambda=args.ssim_lambda,
        alpha_lambda=args.alpha_lambda,
        densify=not args.no_densify,
        random_bkgd=not args.no_random_bkgd,
        device=args.device,
    )
    out = train(cfg)
    print(f"Training finished. Results in {out}")


if __name__ == "__main__":
    main()
