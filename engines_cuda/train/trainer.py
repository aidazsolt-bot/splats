"""CUDA 3D Gaussian Splatting trainer (gsplat + torch) with alpha-aware loss."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset.colmap_rgba import ColmapRGBADataset, init_gaussians_from_points3d  # noqa: E402
from export.ply import export_gsplat_ply  # noqa: E402
from train.losses import alpha_bce, l1_masked, ssim_loss  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from engines_common.resume import (  # noqa: E402
    assert_same_settings,
    load_checkpoint,
    resolve_resume_checkpoint,
    save_checkpoint,
    tensor_to_param,
    write_cfg,
)

# Must match for --resume (same session settings).
CUDA_RESUME_KEYS = (
    "data",
    "data_factor",
    "max_steps",
    "test_every",
    "sh_degree",
    "ssim_lambda",
    "alpha_lambda",
    "densify",
    "random_bkgd",
    "seed",
)


@dataclass
class TrainConfig:
    data: str
    out: str
    data_factor: int = 1
    max_steps: int = 30000
    save_steps: tuple = (7000, 15000, 30000)
    eval_every: int = 1000
    test_every: int = 8
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    ssim_lambda: float = 0.2
    alpha_lambda: float = 0.1
    opacity_reg: float = 0.0
    scale_reg: float = 0.01
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opac_lr: float = 5e-2
    colors_lr: float = 2.5e-3
    batch_size: int = 1
    near: float = 0.01
    far: float = 1.0e10
    densify: bool = True
    densify_start: int = 500
    densify_stop: int = 15000
    densify_every: int = 100
    densify_grad_thresh: float = 0.0002
    densify_scale_thresh: float = 0.01
    opacity_reset_interval: int = 3000
    prune_opa: float = 0.005
    grow_scale2d: float = 0.05
    random_bkgd: bool = True
    alpha_thresh: float = 0.05
    seed: int = 42
    device: str = "cuda"
    resume: bool = False
    ckpt_every: int = 500
    cache_images: bool = False
    cache_workers: int = 8


def _require_cuda(device: str) -> torch.device:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for engines_cuda training, but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch + gsplat on a GPU machine."
            )
        return torch.device(device)
    raise RuntimeError("engines_cuda trainer expects device='cuda' for photorealistic training.")


def _num_sh_bases(degree: int) -> int:
    return (degree + 1) ** 2


class GaussianModel:
    def __init__(self, init, sh_degree: int) -> None:
        self.sh_degree_max = sh_degree
        self.means = torch.nn.Parameter(init.means.clone())
        self.scales = torch.nn.Parameter(init.scales.clone())  # log
        self.quats = torch.nn.Parameter(init.quats.clone())
        self.opacities = torch.nn.Parameter(init.opacities.clone())  # logit
        self.colors_dc = torch.nn.Parameter(init.colors_dc.clone())
        n = init.means.shape[0]
        n_rest = _num_sh_bases(sh_degree) - 1
        if init.colors_rest is not None:
            rest = init.colors_rest
        else:
            rest = torch.zeros(n, n_rest, 3, device=init.means.device, dtype=init.means.dtype)
        self.colors_rest = torch.nn.Parameter(rest)
        self.scene_scale = init.scene_scale

    def parameters_dict(self) -> Dict[str, torch.nn.Parameter]:
        return {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "colors_dc": self.colors_dc,
            "colors_rest": self.colors_rest,
        }

    def colors_sh(self, active_degree: int) -> torch.Tensor:
        """(N, K, 3) SH coeffs with K=(d+1)^2."""
        k = _num_sh_bases(active_degree)
        dc = self.colors_dc[:, None, :]  # N 1 3
        if k == 1:
            return dc
        rest = self.colors_rest[:, : k - 1, :]
        return torch.cat([dc, rest], dim=1)


def _build_optimizers(model: GaussianModel, cfg: TrainConfig, scene_scale: float):
    # Scale means LR by scene extent like original 3DGS
    means_lr = cfg.means_lr * scene_scale
    return {
        "means": torch.optim.Adam([model.means], lr=means_lr, eps=1e-15),
        "scales": torch.optim.Adam([model.scales], lr=cfg.scales_lr, eps=1e-15),
        "quats": torch.optim.Adam([model.quats], lr=cfg.quats_lr, eps=1e-15),
        "opacities": torch.optim.Adam([model.opacities], lr=cfg.opac_lr, eps=1e-15),
        "colors_dc": torch.optim.Adam([model.colors_dc], lr=cfg.colors_lr, eps=1e-15),
        "colors_rest": torch.optim.Adam([model.colors_rest], lr=cfg.colors_lr / 20.0, eps=1e-15),
    }


def _rasterize(model: GaussianModel, viewmats, Ks, width, height, sh_degree_active, near, far):
    from gsplat.rendering import rasterization

    means = model.means
    quats = F.normalize(model.quats, dim=-1)
    scales = torch.exp(model.scales)
    opacities = torch.sigmoid(model.opacities)
    colors = model.colors_sh(sh_degree_active)

    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        sh_degree=sh_degree_active,
        render_mode="RGB",
        packed=False,
    )
    # render_colors: B H W C, render_alphas: B H W 1
    return render_colors, render_alphas.squeeze(-1), info


@torch.no_grad()
def _simple_densify_and_prune(
    model: GaussianModel,
    optimizers: dict,
    grad_means: torch.Tensor,
    cfg: TrainConfig,
    scene_scale: float,
) -> None:
    """Lightweight densify/prune matching classic 3DGS heuristics (no gsplat Strategy API)."""
    grads = grad_means.norm(dim=-1)
    scales = torch.exp(model.scales).max(dim=-1).values

    # Split large + high-grad; clone small + high-grad (snapshot indices first)
    high = grads >= cfg.densify_grad_thresh
    split_mask = high & (scales > cfg.densify_scale_thresh * scene_scale)
    clone_mask = high & ~split_mask
    clone_idx = clone_mask.nonzero(as_tuple=False).squeeze(-1)
    split_idx = split_mask.nonzero(as_tuple=False).squeeze(-1)

    def _append(name: str, new: torch.Tensor) -> None:
        p = getattr(model, name)
        new_p = torch.nn.Parameter(torch.cat([p.data, new], dim=0))
        setattr(model, name, new_p)
        optimizers[name] = torch.optim.Adam(
            [new_p],
            lr=optimizers[name].param_groups[0]["lr"],
            eps=1e-15,
        )

    n_before = model.means.shape[0]

    if clone_idx.numel() > 0:
        _append("means", model.means.data[clone_idx].clone())
        _append("scales", model.scales.data[clone_idx].clone())
        _append("quats", model.quats.data[clone_idx].clone())
        _append("opacities", model.opacities.data[clone_idx].clone())
        _append("colors_dc", model.colors_dc.data[clone_idx].clone())
        _append("colors_rest", model.colors_rest.data[clone_idx].clone())

    if split_idx.numel() > 0:
        n = split_idx.numel()
        std = torch.exp(model.scales.data[split_idx])
        samples = torch.randn(n, 3, device=model.means.device) * std
        means_new = model.means.data[split_idx] + samples
        scales_new = model.scales.data[split_idx] - math.log(1.6)
        model.scales.data[split_idx] = scales_new
        _append("means", means_new)
        _append("scales", scales_new.clone())
        _append("quats", model.quats.data[split_idx].clone())
        _append("opacities", model.opacities.data[split_idx].clone())
        _append("colors_dc", model.colors_dc.data[split_idx].clone())
        _append("colors_rest", model.colors_rest.data[split_idx].clone())

    # Prune low opacity
    opac = torch.sigmoid(model.opacities)
    keep = opac >= cfg.prune_opa
    if keep.sum() < keep.numel():
        for name in ("means", "scales", "quats", "opacities", "colors_dc", "colors_rest"):
            p = getattr(model, name)
            new_p = torch.nn.Parameter(p.data[keep].clone())
            setattr(model, name, new_p)
            optimizers[name] = torch.optim.Adam(
                [new_p],
                lr=optimizers[name].param_groups[0]["lr"],
                eps=1e-15,
            )

    n_after = model.means.shape[0]
    print(
        f"  densify/prune: {n_before} -> {n_after} "
        f"(clone={int(clone_idx.numel())} split={int(split_idx.numel())})"
    )


def _cfg_dict(cfg: TrainConfig) -> dict:
    d = asdict(cfg)
    d["save_steps"] = list(cfg.save_steps)
    return d


def _pack_ckpt(cfg: TrainConfig, model: GaussianModel, step: int, optimizers: dict) -> dict:
    return {
        "step": step,
        "pipeline": "engines_cuda",
        "means": model.means.detach().cpu(),
        "scales": model.scales.detach().cpu(),
        "quats": model.quats.detach().cpu(),
        "opacities": model.opacities.detach().cpu(),
        "colors_dc": model.colors_dc.detach().cpu(),
        "colors_rest": model.colors_rest.detach().cpu(),
        "scene_scale": float(model.scene_scale),
        "optimizers": {k: opt.state_dict() for k, opt in optimizers.items()},
        "cfg": _cfg_dict(cfg),
    }


def _save_cuda_ckpt(
    out: Path, cfg: TrainConfig, model: GaussianModel, step: int, optimizers: dict, named: bool
) -> None:
    payload = _pack_ckpt(cfg, model, step, optimizers)
    also = out / f"ckpt_{step:05d}.pt" if named else None
    save_checkpoint(out, payload, also_named=also)
    if named:
        ply = out / f"splat_{step:05d}.ply"
        export_gsplat_ply(
            ply,
            model.means,
            model.scales,
            F.normalize(model.quats, dim=-1),
            model.opacities,
            model.colors_dc,
            model.colors_rest,
        )
        print(f"  saved {ply} + checkpoint")
    else:
        print(f"  checkpoint step={step} -> {out / 'ckpt_latest.pt'}")


def train(cfg: TrainConfig) -> Path:
    print(f"[startup] Checking device {cfg.device}...", flush=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _require_cuda(cfg.device)
    if device.type == "cuda":
        print(f"[startup] Using {device}: {torch.cuda.get_device_name(device)}", flush=True)

    print("[startup] Importing gsplat...", flush=True)
    try:
        import gsplat  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "gsplat is required. On a CUDA machine: pip install gsplat "
            "(after installing matching torch+cuda). Original error: " + str(e)
        ) from e

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_dict = _cfg_dict(cfg)

    print(f"[startup] Loading COLMAP dataset from {cfg.data}...", flush=True)
    train_ds = ColmapRGBADataset(
        Path(cfg.data), data_factor=cfg.data_factor, split="train", test_every=cfg.test_every
    )
    test_ds = ColmapRGBADataset(
        Path(cfg.data), data_factor=cfg.data_factor, split="test", test_every=cfg.test_every
    )
    if len(train_ds) == 0:
        raise RuntimeError("No training views - check test_every / dataset")
    if cfg.cache_images:
        print(
            f"[startup] Caching images in RAM (factor={cfg.data_factor}, "
            f"workers={cfg.cache_workers})...",
            flush=True,
        )
        train_ds.preload(workers=cfg.cache_workers, label="train")
        test_ds.preload(workers=cfg.cache_workers, label="test")
    else:
        print("[startup] Image cache off: loading one image from disk per step", flush=True)

    start_step = 0
    ckpt_path = resolve_resume_checkpoint(out, cfg.resume)
    if ckpt_path is not None:
        print(f"[startup] Loading checkpoint {ckpt_path}...", flush=True)
        saved = load_checkpoint(ckpt_path, map_location="cpu")
        saved_cfg = saved.get("cfg")
        if saved_cfg is None:
            from engines_common.resume import load_cfg

            saved_cfg = load_cfg(out)
        assert_same_settings(saved_cfg, cfg_dict, CUDA_RESUME_KEYS)
        start_step = int(saved["step"])
        if start_step >= cfg.max_steps:
            print(f"Already finished at step={start_step}; nothing to resume")
            return out
        model = GaussianModel.__new__(GaussianModel)
        model.sh_degree_max = cfg.sh_degree
        model.means = tensor_to_param(saved["means"], device)
        model.scales = tensor_to_param(saved["scales"], device)
        model.quats = tensor_to_param(saved["quats"], device)
        model.opacities = tensor_to_param(saved["opacities"], device)
        model.colors_dc = tensor_to_param(saved["colors_dc"], device)
        model.colors_rest = tensor_to_param(saved["colors_rest"], device)
        model.scene_scale = float(saved.get("scene_scale", 1.0))
        optimizers = _build_optimizers(model, cfg, model.scene_scale)
        opt_state = saved.get("optimizers")
        if opt_state:
            for k, opt in optimizers.items():
                if k in opt_state:
                    try:
                        opt.load_state_dict(opt_state[k])
                    except Exception as e:
                        print(f"  warn: optimizer[{k}] state skipped ({e})")
        print(f"Resume from step={start_step} n={model.means.shape[0]} ({ckpt_path.name})")
    else:
        write_cfg(out, cfg_dict)
        print("[startup] Initializing Gaussians from COLMAP points...", flush=True)
        init = init_gaussians_from_points3d(train_ds.points, device=device, sh_degree=cfg.sh_degree)
        model = GaussianModel(init, sh_degree=cfg.sh_degree)
        optimizers = _build_optimizers(model, cfg, init.scene_scale)

    # Means LR scheduler like 3DGS (exponential decay)
    means_gamma = 0.01 ** (1.0 / max(cfg.max_steps, 1))
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=means_gamma
    )
    if start_step > 0:
        means_sched.last_epoch = start_step
        base_lr = cfg.means_lr * model.scene_scale
        for pg in optimizers["means"].param_groups:
            pg["lr"] = base_lr * (means_gamma ** start_step)

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    it = iter(loader)

    # Running mean of view-space positional gradients for densify
    grad_accum = torch.zeros(model.means.shape[0], 3, device=device)
    grad_count = torch.zeros(model.means.shape[0], device=device)

    t0 = time.time()
    last_log = time.monotonic()
    step = start_step
    print(
        f"Train start: views={len(train_ds)} test={len(test_ds)} "
        f"init_gauss={model.means.shape[0]} factor={cfg.data_factor} "
        f"steps={start_step + 1}..{cfg.max_steps}",
        flush=True,
    )

    try:
        for step in range(start_step + 1, cfg.max_steps + 1):
            if step == start_step + 1:
                print("[startup] Loading first training image...", flush=True)
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            image = batch["image"].to(device)  # B H W 3
            alpha = batch["alpha"].to(device)  # B H W
            K = batch["K"].to(device)
            viewmat = batch["viewmat"].to(device)
            h = int(batch["height"][0])
            w = int(batch["width"][0])

            active_sh = min(cfg.sh_degree, step // cfg.sh_degree_interval)

            # Ensure means require grad for densify stats
            model.means.requires_grad_(True)

            if step == start_step + 1:
                print(
                    "[startup] First gsplat rasterization: CUDA extension may be "
                    "compiled on the CPU before GPU work starts...",
                    flush=True,
                )
            colors, alphas, info = _rasterize(
                model, viewmat, K, w, h, active_sh, cfg.near, cfg.far
            )
            if step == start_step + 1:
                print("[startup] Rasterization returned; computing loss and gradients...", flush=True)

            # Background: random or black behind alpha
            if cfg.random_bkgd:
                bkgd = torch.rand(colors.shape[0], 1, 1, 3, device=device)
            else:
                bkgd = torch.zeros(colors.shape[0], 1, 1, 3, device=device)
            # Composite GT onto same background for fair loss
            gt = image * alpha[..., None] + bkgd * (1.0 - alpha[..., None])
            pred = colors[..., :3] * alphas[..., None] + bkgd * (1.0 - alphas[..., None])

            mask = (alpha > cfg.alpha_thresh).float()
            loss_l1 = l1_masked(pred, gt, mask)
            loss_ssim = ssim_loss(pred, gt, mask)
            loss_a = alpha_bce(alphas, alpha)
            loss = (1.0 - cfg.ssim_lambda) * loss_l1 + cfg.ssim_lambda * loss_ssim
            loss = loss + cfg.alpha_lambda * loss_a
            if cfg.opacity_reg > 0:
                loss = loss + cfg.opacity_reg * torch.sigmoid(model.opacities).mean()
            if cfg.scale_reg > 0:
                loss = loss + cfg.scale_reg * torch.exp(model.scales).mean()

            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            loss.backward()

            # Accumulate densify grads
            if model.means.grad is not None:
                g = model.means.grad.detach()
                # resize accum if densified
                if grad_accum.shape[0] != g.shape[0]:
                    grad_accum = torch.zeros_like(g)
                    grad_count = torch.zeros(g.shape[0], device=device)
                grad_accum += g.abs()
                grad_count += 1.0

            for opt in optimizers.values():
                opt.step()
            means_sched.step()

            # Densify / prune
            if (
                cfg.densify
                and cfg.densify_start <= step <= cfg.densify_stop
                and step % cfg.densify_every == 0
            ):
                avg = grad_accum / grad_count.clamp_min(1.0).unsqueeze(-1)
                _simple_densify_and_prune(model, optimizers, avg, cfg, model.scene_scale)
                grad_accum = torch.zeros(model.means.shape[0], 3, device=device)
                grad_count = torch.zeros(model.means.shape[0], device=device)
                # rebuild means scheduler on new optimizer
                means_sched = torch.optim.lr_scheduler.ExponentialLR(
                    optimizers["means"],
                    gamma=0.01 ** (1.0 / max(cfg.max_steps - step, 1)),
                )

            if cfg.opacity_reset_interval > 0 and step % cfg.opacity_reset_interval == 0 and step < cfg.densify_stop:
                with torch.no_grad():
                    model.opacities.data.clamp_(max=math.log(0.01 / 0.99))

            if (
                step % 100 == 0
                or step == start_step + 1
                or step == cfg.max_steps
                or time.monotonic() - last_log >= 10.0
            ):
                elapsed = time.time() - t0
                print(
                    f"step={step}/{cfg.max_steps} loss={loss.item():.4f} "
                    f"l1={loss_l1.item():.4f} ssim={loss_ssim.item():.4f} "
                    f"alpha={loss_a.item():.4f} n={model.means.shape[0]} "
                    f"sh={active_sh} ({elapsed:.1f}s)",
                    flush=True,
                )
                last_log = time.monotonic()

            named = step in cfg.save_steps or step == cfg.max_steps
            periodic = cfg.ckpt_every > 0 and step % cfg.ckpt_every == 0
            if named or periodic:
                _save_cuda_ckpt(out, cfg, model, step, optimizers, named=named)

            if cfg.eval_every > 0 and step % cfg.eval_every == 0 and len(test_ds) > 0:
                _eval_psnr(model, test_ds, device, cfg, active_sh)
    except KeyboardInterrupt:
        print("Interrupted - writing ckpt_latest.pt (use --resume)")
        _save_cuda_ckpt(out, cfg, model, step, optimizers, named=False)
        raise SystemExit(130)

    meta = {
        "elapsed_sec": round(time.time() - t0, 2),
        "final_gaussians": int(model.means.shape[0]),
        "out": str(out),
        "last_step": cfg.max_steps,
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


@torch.no_grad()
def _eval_psnr(model, test_ds, device, cfg: TrainConfig, active_sh: int) -> None:
    psnrs = []
    for i in range(len(test_ds)):
        sample = test_ds[i]
        image = sample["image"][None].to(device)
        alpha = sample["alpha"][None].to(device)
        K = sample["K"][None].to(device)
        viewmat = sample["viewmat"][None].to(device)
        h = int(sample["height"])
        w = int(sample["width"])
        colors, alphas, _ = _rasterize(model, viewmat, K, w, h, active_sh, cfg.near, cfg.far)
        bkgd = torch.zeros(1, 1, 1, 3, device=device)
        gt = image * alpha[..., None] + bkgd * (1.0 - alpha[..., None])
        pred = colors[..., :3] * alphas[..., None] + bkgd * (1.0 - alphas[..., None])
        mask = alpha > cfg.alpha_thresh
        mse = ((pred - gt) ** 2)[mask.unsqueeze(-1).expand_as(pred)].mean()
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
        psnrs.append(float(psnr))
    print(f"  eval PSNR(mask)={float(np.mean(psnrs)):.2f}dB over {len(psnrs)} views")
