"""CPU quality-first 3DGS trainer (pure PyTorch rasterizer, no CUDA)."""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
for p in (_ROOT, _REPO / "engines_cuda", _REPO / "engines"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset.colmap_rgba import ColmapRGBADataset, init_gaussians_from_points3d  # noqa: E402
from export.ply import export_gsplat_ply  # noqa: E402
from raster.diff_gauss_cpu import rasterize_gaussians_cpu  # noqa: E402
from train.losses import OptionalLPIPS, alpha_bce, l1_masked, ssim_loss  # noqa: E402

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from engines_common.resume import (  # noqa: E402
    assert_same_settings,
    find_checkpoint,
    load_checkpoint,
    save_checkpoint,
    tensor_to_param,
    write_cfg,
)

CPU_RESUME_KEYS = (
    "data",
    "stages",
    "test_every",
    "ssim_lambda",
    "lpips_lambda",
    "alpha_lambda",
    "seed",
    "sh_degree_max",
)


@dataclass
class Stage:
    data_factor: int
    steps: int
    max_gaussians_draw: int
    max_radius: int
    sh_degree: int


@dataclass
class TrainConfig:
    data: str
    out: str
    # Progressive quality schedule (coarse -> fine). More photometric fidelity than
    # typical CUDA factor=4 / 30k shortcuts: ends at higher res + LPIPS + more steps.
    stages: List[Stage] = field(
        default_factory=lambda: [
            Stage(data_factor=8, steps=2000, max_gaussians_draw=12000, max_radius=20, sh_degree=0),
            Stage(data_factor=4, steps=8000, max_gaussians_draw=20000, max_radius=24, sh_degree=1),
            Stage(data_factor=2, steps=20000, max_gaussians_draw=30000, max_radius=28, sh_degree=2),
        ]
    )
    test_every: int = 8
    ssim_lambda: float = 0.2
    lpips_lambda: float = 0.05
    alpha_lambda: float = 0.15
    scale_reg: float = 0.005
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opac_lr: float = 5e-2
    colors_lr: float = 2.5e-3
    densify_every: int = 200
    densify_grad_thresh: float = 0.00015
    densify_scale_thresh: float = 0.01
    prune_opa: float = 0.005
    densify_until_frac: float = 0.7
    random_bkgd: bool = True
    alpha_thresh: float = 0.05
    seed: int = 42
    device: str = "cpu"
    sh_degree_max: int = 2
    save_every_stage: bool = True
    resume: bool = False
    ckpt_every: int = 50


class GaussianModel:
    def __init__(self, init, sh_degree_max: int) -> None:
        self.sh_degree_max = sh_degree_max
        self.means = torch.nn.Parameter(init.means.clone())
        self.scales = torch.nn.Parameter(init.scales.clone())
        self.quats = torch.nn.Parameter(init.quats.clone())
        self.opacities = torch.nn.Parameter(init.opacities.clone())
        self.colors_dc = torch.nn.Parameter(init.colors_dc.clone())
        n = init.means.shape[0]
        n_rest = (sh_degree_max + 1) ** 2 - 1
        rest = (
            init.colors_rest
            if init.colors_rest is not None
            else torch.zeros(n, n_rest, 3, device=init.means.device, dtype=init.means.dtype)
        )
        if rest.shape[1] < n_rest:
            pad = torch.zeros(n, n_rest - rest.shape[1], 3, device=rest.device, dtype=rest.dtype)
            rest = torch.cat([rest, pad], dim=1)
        self.colors_rest = torch.nn.Parameter(rest.clone())
        self.scene_scale = init.scene_scale


def _optimizers(model: GaussianModel, cfg: TrainConfig) -> Dict[str, torch.optim.Optimizer]:
    lr_m = cfg.means_lr * model.scene_scale
    return {
        "means": torch.optim.Adam([model.means], lr=lr_m, eps=1e-15),
        "scales": torch.optim.Adam([model.scales], lr=cfg.scales_lr, eps=1e-15),
        "quats": torch.optim.Adam([model.quats], lr=cfg.quats_lr, eps=1e-15),
        "opacities": torch.optim.Adam([model.opacities], lr=cfg.opac_lr, eps=1e-15),
        "colors_dc": torch.optim.Adam([model.colors_dc], lr=cfg.colors_lr, eps=1e-15),
        "colors_rest": torch.optim.Adam([model.colors_rest], lr=cfg.colors_lr / 20.0, eps=1e-15),
    }


@torch.no_grad()
def _densify_prune(model, optimizers, grad_means, cfg: TrainConfig) -> None:
    grads = grad_means.norm(dim=-1)
    scales = torch.exp(model.scales).max(dim=-1).values
    high = grads >= cfg.densify_grad_thresh
    split_mask = high & (scales > cfg.densify_scale_thresh * model.scene_scale)
    clone_mask = high & ~split_mask
    clone_idx = clone_mask.nonzero(as_tuple=False).squeeze(-1)
    split_idx = split_mask.nonzero(as_tuple=False).squeeze(-1)

    def append(name: str, new: torch.Tensor) -> None:
        p = getattr(model, name)
        np_ = torch.nn.Parameter(torch.cat([p.data, new], dim=0))
        setattr(model, name, np_)
        optimizers[name] = torch.optim.Adam([np_], lr=optimizers[name].param_groups[0]["lr"], eps=1e-15)

    n0 = model.means.shape[0]
    if clone_idx.numel():
        for name in ("means", "scales", "quats", "opacities", "colors_dc", "colors_rest"):
            append(name, getattr(model, name).data[clone_idx].clone())
    if split_idx.numel():
        std = torch.exp(model.scales.data[split_idx])
        means_new = model.means.data[split_idx] + torch.randn_like(std) * std
        scales_new = model.scales.data[split_idx] - math.log(1.6)
        model.scales.data[split_idx] = scales_new
        append("means", means_new)
        append("scales", scales_new.clone())
        for name in ("quats", "opacities", "colors_dc", "colors_rest"):
            append(name, getattr(model, name).data[split_idx].clone())

    keep = torch.sigmoid(model.opacities) >= cfg.prune_opa
    if int(keep.sum()) < keep.numel():
        for name in ("means", "scales", "quats", "opacities", "colors_dc", "colors_rest"):
            p = getattr(model, name)
            np_ = torch.nn.Parameter(p.data[keep].clone())
            setattr(model, name, np_)
            optimizers[name] = torch.optim.Adam([np_], lr=optimizers[name].param_groups[0]["lr"], eps=1e-15)
    print(f"  densify/prune {n0} -> {model.means.shape[0]}")


def _cfg_dump(cfg: TrainConfig) -> dict:
    d = asdict(cfg)
    d["stages"] = [asdict(s) for s in cfg.stages]
    return d


def _pack_cpu_ckpt(
    cfg: TrainConfig,
    model: GaussianModel,
    *,
    stage_i: int,
    local_step: int,
    global_step: int,
    optimizers: dict,
) -> dict:
    return {
        "pipeline": "engines_cpu_train",
        "stage_i": stage_i,
        "local_step": local_step,
        "global_step": global_step,
        "step": global_step,
        "means": model.means.detach().cpu(),
        "scales": model.scales.detach().cpu(),
        "quats": model.quats.detach().cpu(),
        "opacities": model.opacities.detach().cpu(),
        "colors_dc": model.colors_dc.detach().cpu(),
        "colors_rest": model.colors_rest.detach().cpu(),
        "scene_scale": float(model.scene_scale),
        "optimizers": {k: opt.state_dict() for k, opt in optimizers.items()},
        "cfg": _cfg_dump(cfg),
    }


def _load_model_from_ckpt(saved: dict, device: torch.device, sh_degree_max: int) -> GaussianModel:
    model = GaussianModel.__new__(GaussianModel)
    model.sh_degree_max = sh_degree_max
    model.means = tensor_to_param(saved["means"], device)
    model.scales = tensor_to_param(saved["scales"], device)
    model.quats = tensor_to_param(saved["quats"], device)
    model.opacities = tensor_to_param(saved["opacities"], device)
    model.colors_dc = tensor_to_param(saved["colors_dc"], device)
    model.colors_rest = tensor_to_param(saved["colors_rest"], device)
    model.scene_scale = float(saved.get("scene_scale", 1.0))
    return model


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable - falling back to CPU")
        device = torch.device("cpu")

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_dump = _cfg_dump(cfg)

    resume_stage_i = 0
    resume_local_step = 0
    global_step = 0

    if cfg.resume:
        saved = load_checkpoint(find_checkpoint(out), map_location="cpu")
        saved_cfg = saved.get("cfg")
        if saved_cfg is None:
            from engines_common.resume import load_cfg

            saved_cfg = load_cfg(out)
        assert_same_settings(saved_cfg, cfg_dump, CPU_RESUME_KEYS)
        model = _load_model_from_ckpt(saved, device, cfg.sh_degree_max)
        optimizers = _optimizers(model, cfg)
        opt_state = saved.get("optimizers")
        if opt_state:
            for k, opt in optimizers.items():
                if k in opt_state:
                    try:
                        opt.load_state_dict(opt_state[k])
                    except Exception as e:
                        print(f"  warn: optimizer[{k}] state skipped ({e})")
        resume_stage_i = int(saved.get("stage_i", 0))
        resume_local_step = int(saved.get("local_step", 0))
        global_step = int(saved.get("global_step", 0))
        print(
            f"Resume stage={resume_stage_i + 1} local_step={resume_local_step} "
            f"gstep={global_step} n={model.means.shape[0]}"
        )
    else:
        write_cfg(out, cfg_dump)
        probe = ColmapRGBADataset(
            Path(cfg.data), data_factor=8, split="train", test_every=cfg.test_every
        )
        init = init_gaussians_from_points3d(
            probe.points, device=device, sh_degree=cfg.sh_degree_max
        )
        model = GaussianModel(init, sh_degree_max=cfg.sh_degree_max)
        optimizers = _optimizers(model, cfg)

    lpips = OptionalLPIPS()
    if lpips.available:
        print("LPIPS enabled (perceptual loss)")
    else:
        print("LPIPS not installed - continuing with L1+SSIM only (pip install lpips)")

    t0 = time.time()
    total_steps = sum(s.steps for s in cfg.stages)
    stage_i = resume_stage_i
    local_step = resume_local_step

    try:
        for stage_i, stage in enumerate(cfg.stages):
            if stage_i < resume_stage_i:
                continue
            start_local = resume_local_step + 1 if stage_i == resume_stage_i else 1
            if start_local > stage.steps:
                continue
            print(
                f"=== Stage {stage_i+1}/{len(cfg.stages)}: factor={stage.data_factor} "
                f"steps={start_local}..{stage.steps} sh<={stage.sh_degree} "
                f"draw<={stage.max_gaussians_draw}"
            )
            train_ds = ColmapRGBADataset(
                Path(cfg.data),
                data_factor=stage.data_factor,
                split="train",
                test_every=cfg.test_every,
            )
            densify_until = int(stage.steps * cfg.densify_until_frac)
            grad_accum = torch.zeros_like(model.means)
            grad_count = torch.zeros(model.means.shape[0], device=device)

            for local_step in range(start_local, stage.steps + 1):
                global_step += 1
                sample = train_ds[random.randrange(len(train_ds))]
                image = sample["image"].to(device)
                alpha = sample["alpha"].to(device)
                K = sample["K"].to(device)
                viewmat = sample["viewmat"].to(device)
                h, w = int(sample["height"]), int(sample["width"])

                model.means.requires_grad_(True)
                rgb, alph = rasterize_gaussians_cpu(
                    model.means,
                    model.scales,
                    model.quats,
                    model.opacities,
                    model.colors_dc,
                    model.colors_rest,
                    viewmat,
                    K,
                    w,
                    h,
                    sh_degree=stage.sh_degree,
                    max_gaussians=stage.max_gaussians_draw,
                    max_radius=stage.max_radius,
                )

                if cfg.random_bkgd:
                    bkgd = torch.rand(1, 1, 3, device=device)
                else:
                    bkgd = torch.zeros(1, 1, 3, device=device)
                gt = image * alpha[..., None] + bkgd * (1.0 - alpha[..., None])
                pred = rgb * alph[..., None] + bkgd * (1.0 - alph[..., None])
                mask = (alpha > cfg.alpha_thresh).float()

                loss_l1 = l1_masked(pred, gt, mask)
                loss_s = ssim_loss(pred, gt, mask)
                loss_a = alpha_bce(alph, alpha)
                loss = (1.0 - cfg.ssim_lambda) * loss_l1 + cfg.ssim_lambda * loss_s
                loss = loss + cfg.alpha_lambda * loss_a
                if cfg.lpips_lambda > 0 and lpips.available and local_step % 2 == 0:
                    loss = loss + cfg.lpips_lambda * lpips(pred, gt, mask)
                if cfg.scale_reg > 0:
                    loss = loss + cfg.scale_reg * torch.exp(model.scales).mean()

                for opt in optimizers.values():
                    opt.zero_grad(set_to_none=True)
                loss.backward()
                if model.means.grad is not None:
                    g = model.means.grad.detach()
                    if grad_accum.shape[0] != g.shape[0]:
                        grad_accum = torch.zeros_like(g)
                        grad_count = torch.zeros(g.shape[0], device=device)
                    grad_accum += g.abs()
                    grad_count += 1
                for opt in optimizers.values():
                    opt.step()

                if (
                    local_step <= densify_until
                    and cfg.densify_every > 0
                    and local_step % cfg.densify_every == 0
                ):
                    avg = grad_accum / grad_count.clamp_min(1).unsqueeze(-1)
                    _densify_prune(model, optimizers, avg, cfg)
                    grad_accum = torch.zeros_like(model.means)
                    grad_count = torch.zeros(model.means.shape[0], device=device)

                if local_step % 50 == 0 or local_step == 1:
                    print(
                        f"gstep={global_step}/{total_steps} stage={stage_i+1} "
                        f"loc={local_step}/{stage.steps} loss={loss.item():.4f} "
                        f"l1={loss_l1.item():.4f} n={model.means.shape[0]} "
                        f"({time.time()-t0:.0f}s)"
                    )

                if cfg.ckpt_every > 0 and local_step % cfg.ckpt_every == 0:
                    save_checkpoint(
                        out,
                        _pack_cpu_ckpt(
                            cfg,
                            model,
                            stage_i=stage_i,
                            local_step=local_step,
                            global_step=global_step,
                            optimizers=optimizers,
                        ),
                    )
                    print(f"  checkpoint stage={stage_i+1} loc={local_step}")

            if cfg.save_every_stage:
                ply = out / f"splat_stage{stage_i+1}_factor{stage.data_factor}.ply"
                export_gsplat_ply(
                    ply,
                    model.means,
                    model.scales,
                    F.normalize(model.quats, dim=-1),
                    model.opacities,
                    model.colors_dc,
                    model.colors_rest,
                )
                save_checkpoint(
                    out,
                    _pack_cpu_ckpt(
                        cfg,
                        model,
                        stage_i=stage_i,
                        local_step=stage.steps,
                        global_step=global_step,
                        optimizers=optimizers,
                    ),
                    also_named=out / f"ckpt_stage{stage_i+1}.pt",
                )
                print(f"  saved {ply}")
            # After finishing a resumed stage, next stages start from local 1
            resume_local_step = 0
    except KeyboardInterrupt:
        print("Interrupted - writing ckpt_latest.pt (use --resume)")
        save_checkpoint(
            out,
            _pack_cpu_ckpt(
                cfg,
                model,
                stage_i=stage_i,
                local_step=local_step,
                global_step=global_step,
                optimizers=optimizers,
            ),
        )
        raise SystemExit(130)

    final_ply = out / "splat_final.ply"
    export_gsplat_ply(
        final_ply,
        model.means,
        model.scales,
        F.normalize(model.quats, dim=-1),
        model.opacities,
        model.colors_dc,
        model.colors_rest,
    )
    meta = {
        "elapsed_sec": round(time.time() - t0, 2),
        "final_gaussians": int(model.means.shape[0]),
        "device": str(device),
        "lpips": lpips.available,
        "out": str(out),
        "global_step": global_step,
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return out
