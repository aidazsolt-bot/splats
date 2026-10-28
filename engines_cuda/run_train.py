#!/usr/bin/env python3
"""Train photorealistic 3D Gaussians with gsplat + PyTorch CUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import win_build_env  # noqa: E402

print("[startup] Preparing build environment...", flush=True)
win_build_env.setup()

print("[startup] Loading training modules...", flush=True)
from train.trainer import TrainConfig, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CUDA/Torch 3DGS trainer (alpha-masked, COLMAP RGBA)"
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("/usr/src/TestAdnroidNDKSDK/blackberry-splats") / "dataset",
        help="COLMAP root with cameras.txt / images.txt / points3D.txt / images/",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/usr/src/TestAdnroidNDKSDK/blackberry-splats") / "engines_cuda_out",
    )
    ap.add_argument(
        "--data-factor",
        type=int,
        default=1,
        help="Downscale training images by this factor (1 = original resolution)",
    )
    ap.add_argument(
        "--cache-images",
        action="store_true",
        help="Load all (downscaled) images into RAM once at startup instead of "
        "reading one image from disk per step",
    )
    ap.add_argument(
        "--cache-workers",
        type=int,
        default=8,
        help="Parallel image loader threads for --cache-images",
    )
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--alpha-lambda", type=float, default=0.1)
    ap.add_argument(
        "--strategy",
        choices=("legacy", "default", "mcmc"),
        default=None,
        help="Gaussian refinement strategy (new runs default to default; old checkpoints resume as legacy)",
    )
    ap.add_argument("--refine-start", type=int)
    ap.add_argument(
        "--refine-stop",
        type=int,
        help="Stop refinement (default: 15000 for default, 25000 for mcmc)",
    )
    ap.add_argument("--refine-every", type=int)
    ap.add_argument("--prune-opa", type=float, default=0.005)
    ap.add_argument("--grow-grad2d", type=float)
    ap.add_argument("--absgrad", action="store_true")
    ap.add_argument("--grow-scale3d", type=float)
    ap.add_argument("--prune-scale3d", type=float)
    ap.add_argument("--reset-every", type=int)
    ap.add_argument("--pause-refine-after-reset", type=int)
    ap.add_argument("--revised-opacity", action="store_true")
    ap.add_argument("--cap-max", type=int, default=200000)
    ap.add_argument("--noise-lr", type=float, default=5e5)
    ap.add_argument(
        "--opacity-reg",
        type=float,
        help="Opacity regularization (default 0.01 for mcmc, otherwise 0)",
    )
    ap.add_argument("--scale-reg", type=float, default=0.01)
    ap.add_argument(
        "--antialiased",
        action="store_true",
        help="Use gsplat's antialiased rasterization mode",
    )
    ap.add_argument("--no-densify", action="store_true")
    ap.add_argument("--no-random-bkgd", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run (500 steps) to verify the stack on a GPU box",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Continue from ckpt_latest.pt in --out (same settings required)",
    )
    ap.add_argument(
        "--ckpt-every",
        type=int,
        default=500,
        help="Write ckpt_latest.pt every N steps (0=only save_steps / interrupt)",
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
        strategy=args.strategy,
        refine_start=args.refine_start,
        refine_stop=args.refine_stop,
        refine_every=args.refine_every,
        prune_opa=args.prune_opa,
        grow_grad2d=args.grow_grad2d,
        absgrad=args.absgrad,
        grow_scale3d=args.grow_scale3d,
        prune_scale3d=args.prune_scale3d,
        reset_every=args.reset_every,
        pause_refine_after_reset=args.pause_refine_after_reset,
        revised_opacity=args.revised_opacity,
        cap_max=args.cap_max,
        noise_lr=args.noise_lr,
        opacity_reg=args.opacity_reg,
        scale_reg=args.scale_reg,
        antialiased=args.antialiased,
        densify=not args.no_densify,
        random_bkgd=not args.no_random_bkgd,
        device=args.device,
        resume=args.resume,
        ckpt_every=args.ckpt_every,
        cache_images=args.cache_images,
        cache_workers=args.cache_workers,
    )
    out = train(cfg)
    print(f"Training finished. Results in {out}")


if __name__ == "__main__":
    main()
