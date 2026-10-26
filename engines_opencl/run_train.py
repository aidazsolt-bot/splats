#!/usr/bin/env python3
"""Train Gaussians with OpenCL forward (hybrid backward on CPU)."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(REPO / "engines_cuda"), str(REPO / "engines_cpu_train"), str(REPO / "engines")]

from dataset.colmap_rgba import ColmapRGBADataset, init_gaussians_from_points3d  # noqa: E402
from export.ply import export_gsplat_ply  # noqa: E402
from raster.cl_rasterizer import OpenCLGaussianRasterizer  # noqa: E402
from train.hybrid_raster import rasterize_hybrid  # noqa: E402


def _losses_from_cpu():
    import importlib.util

    path = REPO / "engines_cpu_train" / "train" / "losses.py"
    spec = importlib.util.spec_from_file_location("cpu_losses", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _cpu_raster():
    import importlib.util

    path = REPO / "engines_cpu_train" / "raster" / "diff_gauss_cpu.py"
    spec = importlib.util.spec_from_file_location("diff_gauss_cpu", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenCL hybrid 3DGS trainer")
    ap.add_argument("--data", type=Path, default=Path.home() / "tmp" / "blackberry-splats" / "dataset")
    ap.add_argument("--out", type=Path, default=REPO / "opencl_out")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--data-factor", type=int, default=4)
    ap.add_argument("--device-index", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--cpu-fallback",
        action="store_true",
        help="If no OpenCL, train with engines_cpu_train rasterizer only",
    )
    args = ap.parse_args()

    losses = _losses_from_cpu()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    steps = 2 if args.smoke else args.steps
    factor = 8 if args.smoke else args.data_factor
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Smoke keeps a tiny draw budget so CPU fallback finishes quickly.

    cl = None
    try:
        cl = OpenCLGaussianRasterizer(device_index=args.device_index)
    except Exception as e:
        print(f"OpenCL init failed: {e}")
        if not args.cpu_fallback:
            raise SystemExit(
                "No OpenCL device. On Intel Arc install compute runtime, or pass --cpu-fallback."
            )
        print("Using CPU rasterizer fallback for the whole step")

    ds = ColmapRGBADataset(Path(args.data), data_factor=factor, split="train", test_every=8)
    init = init_gaussians_from_points3d(ds.points, device=torch.device("cpu"), sh_degree=2)
    means = torch.nn.Parameter(init.means.clone())
    scales = torch.nn.Parameter(init.scales.clone())
    quats = torch.nn.Parameter(init.quats.clone())
    opac = torch.nn.Parameter(init.opacities.clone())
    dc = torch.nn.Parameter(init.colors_dc.clone())
    rest = torch.nn.Parameter(
        init.colors_rest.clone()
        if init.colors_rest is not None
        else torch.zeros(means.shape[0], 8, 3)
    )
    opt = torch.optim.Adam(
        [
            {"params": [means], "lr": 1.6e-4 * init.scene_scale},
            {"params": [scales], "lr": 5e-3},
            {"params": [quats], "lr": 1e-3},
            {"params": [opac], "lr": 5e-2},
            {"params": [dc], "lr": 2.5e-3},
            {"params": [rest], "lr": 2.5e-3 / 20},
        ],
        eps=1e-15,
    )

    t0 = time.time()
    max_draw = 1500 if args.smoke else 20000
    for step in range(1, steps + 1):
        sample = ds[random.randrange(len(ds))]
        image = sample["image"]
        alpha = sample["alpha"]
        K = sample["K"]
        viewmat = sample["viewmat"]
        h, w = int(sample["height"]), int(sample["width"])

        if cl is not None:
            rgb, alph = rasterize_hybrid(
                means,
                scales,
                quats,
                opac,
                dc,
                rest,
                viewmat,
                K,
                w,
                h,
                sh_degree=min(2, step // 500),
                cl_rasterizer=cl,
                max_gaussians=max_draw,
                max_radius=24,
            )
        else:
            rgb, alph = _cpu_raster().rasterize_gaussians_cpu(
                means, scales, quats, opac, dc, rest, viewmat, K, w, h,
                sh_degree=min(2, step // 500), max_gaussians=max_draw, max_radius=24,
            )

        bkgd = torch.rand(1, 1, 3)
        gt = image * alpha[..., None] + bkgd * (1 - alpha[..., None])
        pred = rgb * alph[..., None] + bkgd * (1 - alph[..., None])
        mask = (alpha > 0.05).float()
        loss = (1 - 0.2) * losses.l1_masked(pred, gt, mask) + 0.2 * losses.ssim_loss(pred, gt, mask)
        loss = loss + 0.1 * losses.alpha_bce(alph, alpha)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == 1:
            print(f"step={step}/{steps} loss={loss.item():.4f} n={means.shape[0]} ({time.time()-t0:.0f}s)")

    ply = out / "splat_opencl.ply"
    export_gsplat_ply(ply, means, scales, F.normalize(quats, dim=-1), opac, dc, rest)
    meta = {"elapsed_sec": round(time.time() - t0, 2), "steps": steps, "opencl": cl is not None}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {ply}")
    print(meta)


if __name__ == "__main__":
    main()
