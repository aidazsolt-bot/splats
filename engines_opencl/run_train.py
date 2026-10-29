#!/usr/bin/env python3
"""Train Gaussians with OpenCL forward (hybrid backward on CPU)."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(REPO / "engines_cuda"), str(REPO / "engines_cpu_train"), str(REPO), str(REPO / "engines")]

from dataset.colmap_rgba import ColmapRGBADataset, init_gaussians_from_points3d  # noqa: E402
from engines_common.resume import (  # noqa: E402
    assert_same_settings,
    load_checkpoint,
    resolve_resume_checkpoint,
    save_checkpoint,
    tensor_to_param,
    write_cfg,
)
from export.ply import export_gsplat_ply  # noqa: E402
from raster.cl_rasterizer import OpenCLGaussianRasterizer  # noqa: E402
from train.hybrid_raster import rasterize_hybrid  # noqa: E402

OPENCL_RESUME_KEYS = (
    "data",
    "steps",
    "data_factor",
    "seed",
    "prefer",
    "cpu_fallback",
    "smoke",
)


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


def _pack(
    *,
    step: int,
    means,
    scales,
    quats,
    opac,
    dc,
    rest,
    opt,
    cfg: dict,
    scene_scale: float,
) -> dict:
    return {
        "pipeline": "engines_opencl",
        "step": step,
        "means": means.detach().cpu(),
        "scales": scales.detach().cpu(),
        "quats": quats.detach().cpu(),
        "opacities": opac.detach().cpu(),
        "colors_dc": dc.detach().cpu(),
        "colors_rest": rest.detach().cpu(),
        "scene_scale": float(scene_scale),
        "optimizer": opt.state_dict(),
        "cfg": cfg,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenCL hybrid 3DGS trainer")
    ap.add_argument("--data", type=Path, default=Path("/usr/src/TestAdnroidNDKSDK/blackberry-splats") / "dataset")
    ap.add_argument("--out", type=Path, default=REPO / "opencl_out")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--data-factor", type=int, default=4)
    ap.add_argument("--device-index", type=int, default=None)
    ap.add_argument(
        "--prefer",
        type=str,
        default="nvidia",
        help="OpenCL device name substring (default: nvidia). Use '' for auto.",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--cpu-fallback",
        action="store_true",
        help="If no OpenCL, train with engines_cpu_train rasterizer only",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Continue from ckpt_latest.pt in --out (same settings required)",
    )
    ap.add_argument(
        "--ckpt-every",
        type=int,
        default=50,
        help="Write ckpt_latest.pt every N steps (0=end / interrupt only)",
    )
    ap.add_argument(
        "--ply-every",
        type=int,
        default=100,
        help=(
            "Write numbered splat_XXXXX.ply every N steps (0=final only). "
            "Also refreshes splat_latest.ply on each checkpoint."
        ),
    )
    ap.add_argument(
        "--cpu-batch-size",
        type=int,
        default=64,
        help="CPU raster: Gaussians per stencil batch (batched method)",
    )
    ap.add_argument(
        "--cpu-tile-size",
        type=int,
        default=64,
        help="CPU raster: image tile size; pixels clipped per tile (0=one full-frame pass)",
    )
    ap.add_argument(
        "--cpu-method",
        choices=("batched", "loop"),
        default="batched",
        help="CPU raster backend (batched=fast stencil, loop=legacy)",
    )
    ap.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="torch.set_num_threads for CPU raster (0=auto)",
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

    cfg = {
        "data": str(Path(args.data).expanduser().resolve()),
        "out": str(out.resolve()),
        "steps": steps,
        "data_factor": factor,
        "seed": args.seed,
        "prefer": args.prefer.strip(),
        "cpu_fallback": bool(args.cpu_fallback),
        "smoke": bool(args.smoke),
        "device_index": args.device_index,
        "cpu_batch_size": int(args.cpu_batch_size),
        "cpu_tile_size": int(args.cpu_tile_size),
        "cpu_method": str(args.cpu_method),
        "cpu_threads": int(args.cpu_threads),
    }
    cpu_threads = None if args.cpu_threads <= 0 else int(args.cpu_threads)

    cl = None
    prefer = args.prefer.strip() or None
    try:
        cl = OpenCLGaussianRasterizer(device_index=args.device_index, prefer=prefer)
        print(
            "Note: training uses CPU differentiable rasterizer; "
            "OpenCL device is reserved for no-grad preview paths."
        )
    except Exception as e:
        print(f"OpenCL init failed: {e}")
        if not args.cpu_fallback:
            raise SystemExit(
                "No OpenCL device. Pass --prefer nvidia or --device-index 1, "
                "or --cpu-fallback for pure CPU training."
            )
        print("Using CPU rasterizer fallback for the whole step")

    ds = ColmapRGBADataset(Path(args.data), data_factor=factor, split="train", test_every=8)
    start_step = 0
    device = torch.device("cpu")

    saved = None
    ckpt_path = resolve_resume_checkpoint(out, args.resume)
    if ckpt_path is not None:
        saved = load_checkpoint(ckpt_path, map_location="cpu")
        saved_cfg = saved.get("cfg")
        if saved_cfg is None:
            from engines_common.resume import load_cfg

            saved_cfg = load_cfg(out)
        assert_same_settings(saved_cfg, cfg, OPENCL_RESUME_KEYS)
        start_step = int(saved["step"])
        if start_step >= steps:
            print(f"Already finished at step={start_step}; nothing to resume")
            return
        means = tensor_to_param(saved["means"], device)
        scales = tensor_to_param(saved["scales"], device)
        quats = tensor_to_param(saved["quats"], device)
        opac = tensor_to_param(saved["opacities"], device)
        dc = tensor_to_param(saved["colors_dc"], device)
        rest = tensor_to_param(saved["colors_rest"], device)
        scene_scale = float(saved.get("scene_scale", 1.0))
        print(f"Resume from step={start_step} n={means.shape[0]} ({ckpt_path.name})")
    else:
        write_cfg(out, cfg)
        init = init_gaussians_from_points3d(ds.points, device=device, sh_degree=2)
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
        scene_scale = float(init.scene_scale)

    opt = torch.optim.Adam(
        [
            {"params": [means], "lr": 1.6e-4 * scene_scale},
            {"params": [scales], "lr": 5e-3},
            {"params": [quats], "lr": 1e-3},
            {"params": [opac], "lr": 5e-2},
            {"params": [dc], "lr": 2.5e-3},
            {"params": [rest], "lr": 2.5e-3 / 20},
        ],
        eps=1e-15,
    )
    if saved is not None and "optimizer" in saved:
        try:
            opt.load_state_dict(saved["optimizer"])
        except Exception as e:
            print(f"  warn: optimizer state skipped ({e})")

    t0 = time.time()
    max_draw = 1500 if args.smoke else 20000
    step = start_step
    try:
        for step in range(start_step + 1, steps + 1):
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
                    batch_size=args.cpu_batch_size,
                    tile_size=args.cpu_tile_size,
                    method=args.cpu_method,
                    num_threads=cpu_threads,
                )
            else:
                rgb, alph = _cpu_raster().rasterize_gaussians_cpu(
                    means, scales, quats, opac, dc, rest, viewmat, K, w, h,
                    sh_degree=min(2, step // 500), max_gaussians=max_draw, max_radius=24,
                    batch_size=args.cpu_batch_size,
                    tile_size=args.cpu_tile_size,
                    method=args.cpu_method,
                    num_threads=cpu_threads,
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

            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                save_checkpoint(
                    out,
                    _pack(
                        step=step,
                        means=means,
                        scales=scales,
                        quats=quats,
                        opac=opac,
                        dc=dc,
                        rest=rest,
                        opt=opt,
                        cfg=cfg,
                        scene_scale=scene_scale,
                    ),
                )
                # Cheap viewer snapshot (~0.5s / ~10MB at 38k); overwrite in place.
                export_gsplat_ply(
                    out / "splat_latest.ply",
                    means,
                    scales,
                    F.normalize(quats, dim=-1),
                    opac,
                    dc,
                    rest,
                )
                print(f"  checkpoint step={step}")
                if args.ply_every > 0 and step % args.ply_every == 0:
                    ply_n = out / f"splat_{step:05d}.ply"
                    export_gsplat_ply(
                        ply_n,
                        means,
                        scales,
                        F.normalize(quats, dim=-1),
                        opac,
                        dc,
                        rest,
                    )
                    print(f"  saved {ply_n.name}")
    except KeyboardInterrupt:
        print("Interrupted - writing ckpt_latest.pt (use --resume)")
        save_checkpoint(
            out,
            _pack(
                step=step,
                means=means,
                scales=scales,
                quats=quats,
                opac=opac,
                dc=dc,
                rest=rest,
                opt=opt,
                cfg=cfg,
                scene_scale=scene_scale,
            ),
        )
        raise SystemExit(130)

    ply = out / "splat_opencl.ply"
    export_gsplat_ply(ply, means, scales, F.normalize(quats, dim=-1), opac, dc, rest)
    export_gsplat_ply(
        out / "splat_latest.ply",
        means,
        scales,
        F.normalize(quats, dim=-1),
        opac,
        dc,
        rest,
    )
    save_checkpoint(
        out,
        _pack(
            step=steps,
            means=means,
            scales=scales,
            quats=quats,
            opac=opac,
            dc=dc,
            rest=rest,
            opt=opt,
            cfg=cfg,
            scene_scale=scene_scale,
        ),
        also_named=out / f"ckpt_{steps:05d}.pt",
    )
    meta = {
        "elapsed_sec": round(time.time() - t0, 2),
        "steps": steps,
        "last_step": steps,
        "opencl": cl is not None,
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {ply}")
    print(meta)


if __name__ == "__main__":
    main()
