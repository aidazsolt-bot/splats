"""Torch autograd wrapper: OpenCL forward, CPU differentiable backward."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]


def _cpu_raster_mod():
    import importlib.util

    path = _REPO / "engines_cpu_train" / "raster" / "diff_gauss_cpu.py"
    spec = importlib.util.spec_from_file_location("diff_gauss_cpu", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class HybridOpenCLRaster(torch.autograd.Function):
    """Forward via OpenCL (fast on Intel Arc / AMD). Backward via CPU torch path."""

    @staticmethod
    def forward(
        ctx,
        means,
        scales_log,
        quats,
        opacities_logit,
        colors_dc,
        colors_rest,
        viewmat,
        K,
        width: int,
        height: int,
        sh_degree: int,
        cl_rasterizer,
        max_gaussians: int,
        max_radius: int,
    ):
        ctx.cl = cl_rasterizer
        ctx.width = width
        ctx.height = height
        ctx.sh_degree = sh_degree
        ctx.max_gaussians = max_gaussians
        ctx.max_radius = max_radius
        ctx.save_for_backward(
            means, scales_log, quats, opacities_logit, colors_dc, colors_rest, viewmat, K
        )

        with torch.no_grad():
            cpu_r = _cpu_raster_mod()
            Rcw = viewmat[:3, :3]
            tcw = viewmat[:3, 3]
            means_c = (Rcw @ means.T).T + tcw
            dirs = F.normalize(-means_c, dim=-1)
            rest = colors_rest if colors_rest is not None else None
            rgbs = cpu_r.eval_sh(colors_dc, rest, dirs, sh_degree)
            opac = torch.sigmoid(opacities_logit)

            _, _, depths, _, valid = cpu_r.project_gaussians(
                means, scales_log, quats, viewmat, K, width, height
            )
            idx = valid.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                rgb = means.new_zeros(height, width, 3)
                alph = means.new_zeros(height, width)
                return rgb, alph
            order = torch.argsort(depths[idx])
            idx = idx[order]
            if idx.numel() > max_gaussians:
                idx = idx[:max_gaussians]

            rgb_np, a_np = cl_rasterizer.rasterize(
                means[idx].detach().cpu().numpy(),
                scales_log[idx].detach().cpu().numpy(),
                F.normalize(quats[idx], dim=-1).detach().cpu().numpy(),
                opac[idx].detach().cpu().numpy(),
                rgbs[idx].detach().cpu().numpy(),
                viewmat.detach().cpu().numpy(),
                K.detach().cpu().numpy(),
                width,
                height,
                max_radius=max_radius,
            )
            rgb = torch.from_numpy(rgb_np).to(means.device).to(means.dtype)
            alph = torch.from_numpy(a_np).to(means.device).to(means.dtype)
        return rgb, alph

    @staticmethod
    def backward(ctx, grad_rgb, grad_a):
        (
            means,
            scales_log,
            quats,
            opacities_logit,
            colors_dc,
            colors_rest,
            viewmat,
            K,
        ) = ctx.saved_tensors
        # Recompute with CPU differentiable rasterizer to obtain grads
        means_ = means.detach().requires_grad_(True)
        scales_ = scales_log.detach().requires_grad_(True)
        quats_ = quats.detach().requires_grad_(True)
        opac_ = opacities_logit.detach().requires_grad_(True)
        dc_ = colors_dc.detach().requires_grad_(True)
        rest_ = colors_rest.detach().requires_grad_(True) if colors_rest is not None else None

        cpu_r = _cpu_raster_mod()
        rgb, alph = cpu_r.rasterize_gaussians_cpu(
            means_,
            scales_,
            quats_,
            opac_,
            dc_,
            rest_,
            viewmat.detach(),
            K.detach(),
            ctx.width,
            ctx.height,
            sh_degree=ctx.sh_degree,
            max_gaussians=ctx.max_gaussians,
            max_radius=ctx.max_radius,
        )
        loss = (rgb * grad_rgb).sum() + (alph * grad_a).sum()
        grads = torch.autograd.grad(
            loss,
            [means_, scales_, quats_, opac_, dc_] + ([rest_] if rest_ is not None else []),
            retain_graph=False,
            allow_unused=True,
        )
        g_means, g_scales, g_quats, g_opac, g_dc = grads[:5]
        g_rest = grads[5] if rest_ is not None else None
        return (
            g_means,
            g_scales,
            g_quats,
            g_opac,
            g_dc,
            g_rest,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rasterize_hybrid(
    means,
    scales_log,
    quats,
    opacities_logit,
    colors_dc,
    colors_rest,
    viewmat,
    K,
    width: int,
    height: int,
    sh_degree: int,
    cl_rasterizer,
    max_gaussians: int = 25000,
    max_radius: int = 28,
):
    return HybridOpenCLRaster.apply(
        means,
        scales_log,
        quats,
        opacities_logit,
        colors_dc,
        colors_rest,
        viewmat,
        K,
        width,
        height,
        sh_degree,
        cl_rasterizer,
        max_gaussians,
        max_radius,
    )
