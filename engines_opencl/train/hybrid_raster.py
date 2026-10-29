"""Torch raster helpers: OpenCL for inference, CPU for differentiable training."""

from __future__ import annotations

from pathlib import Path

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
    """OpenCL forward for inference-style calls; backward uses CPU with enable_grad."""

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
                return means.new_zeros(height, width, 3), means.new_zeros(height, width)
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

        # Nested autograd during backward requires enable_grad()
        with torch.enable_grad():
            means_ = means.detach().requires_grad_(True)
            scales_ = scales_log.detach().requires_grad_(True)
            quats_ = quats.detach().requires_grad_(True)
            opac_ = opacities_logit.detach().requires_grad_(True)
            dc_ = colors_dc.detach().requires_grad_(True)
            rest_ = (
                colors_rest.detach().requires_grad_(True) if colors_rest is not None else None
            )
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
                method="batched",
            )
            gr = grad_rgb if grad_rgb is not None else torch.zeros_like(rgb)
            ga = grad_a if grad_a is not None else torch.zeros_like(alph)
            loss = (rgb * gr).sum() + (alph * ga).sum()
            if not loss.requires_grad:
                zeros = [torch.zeros_like(means_)] * 5
                g_rest = torch.zeros_like(rest_) if rest_ is not None else None
                return (*zeros, g_rest, None, None, None, None, None, None, None, None)

            inputs = [means_, scales_, quats_, opac_, dc_]
            if rest_ is not None:
                inputs.append(rest_)
            grads = torch.autograd.grad(loss, inputs, retain_graph=False, allow_unused=True)

        def _g(t, ref):
            return t if t is not None else torch.zeros_like(ref)

        g_means = _g(grads[0], means_)
        g_scales = _g(grads[1], scales_)
        g_quats = _g(grads[2], quats_)
        g_opac = _g(grads[3], opac_)
        g_dc = _g(grads[4], dc_)
        g_rest = _g(grads[5], rest_) if rest_ is not None else None
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
    batch_size: int = 64,
    tile_size: int = 64,
    method: str = "batched",
    num_threads: int | None = None,
):
    """During training (params require grad) use CPU differentiable rasterizer.

    OpenCL forward is used only when no grad is needed (faster preview).
    """
    needs_grad = any(
        (t is not None and getattr(t, "requires_grad", False))
        for t in (means, scales_log, quats, opacities_logit, colors_dc, colors_rest)
    )
    if needs_grad or cl_rasterizer is None:
        cpu_r = _cpu_raster_mod()
        return cpu_r.rasterize_gaussians_cpu(
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
            sh_degree=sh_degree,
            max_gaussians=max_gaussians,
            max_radius=max_radius,
            batch_size=batch_size,
            tile_size=tile_size,
            method=method,
            num_threads=num_threads,
        )
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
