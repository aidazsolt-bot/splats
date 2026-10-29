"""Differentiable 3D Gaussian rasterizer in pure PyTorch (CPU-first).

Uses depth-weighted alpha accumulation (commutative sum, no sequential
transmittance), so gradients flow through means/scales/quats/opacity/color.

Hot path: batched stencil scatter (not a Python loop over every Gaussian).
Optional image tiling for cache locality. Torch intra-op threads for matmul-like
ops; avoid threaded autograd (not thread-safe).
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quats, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).view(-1, 3, 3)


def project_gaussians(
    means: torch.Tensor,
    scales_log: torch.Tensor,
    quats: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    near: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Rcw = viewmat[:3, :3]
    tcw = viewmat[:3, 3]
    means_c = (Rcw @ means.T).T + tcw
    depths = means_c[:, 2]
    valid = depths > near

    scales = torch.exp(scales_log)
    R = quat_to_rotmat(quats)
    S = torch.diag_embed(scales)
    cov_w = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
    cov_c = Rcw @ cov_w @ Rcw.transpose(-1, -2)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x, y, z = means_c[:, 0], means_c[:, 1], means_c[:, 2].clamp_min(near)
    means2d = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)

    z2 = (z * z).clamp_min(1e-8)
    J = means.new_zeros(means.shape[0], 2, 3)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * x / z2
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * y / z2
    cov2d = J @ cov_c @ J.transpose(-1, -2)
    eye = torch.eye(2, device=means.device, dtype=means.dtype).expand(cov2d.shape[0], -1, -1)
    cov2d = cov2d + 0.3 * eye

    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    mid = 0.5 * (a + c)
    disc = torch.sqrt(((0.5 * (a - c)) ** 2 + b * b).clamp_min(0) + 1e-12)
    lmax = mid + disc
    radii = (3.0 * torch.sqrt(lmax.clamp_min(1e-8))).ceil()
    max_r = float(max(width, height))
    valid = (
        valid
        & (means2d[:, 0] + radii >= 0)
        & (means2d[:, 0] - radii < width)
        & (means2d[:, 1] + radii >= 0)
        & (means2d[:, 1] - radii < height)
        & (radii > 0.5)
        & (radii < max_r)
    )
    return means2d, cov2d, depths, radii, valid


def eval_sh(
    colors_dc: torch.Tensor,
    colors_rest: Optional[torch.Tensor],
    dirs: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    c0 = 0.28209479177387814
    rgb = c0 * colors_dc
    if degree < 1 or colors_rest is None or colors_rest.shape[1] < 3:
        return (0.5 + rgb).clamp(0, 1)
    c1 = 0.4886025119029199
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    rgb = rgb + c1 * (
        -y[:, None] * colors_rest[:, 0]
        + z[:, None] * colors_rest[:, 1]
        - x[:, None] * colors_rest[:, 2]
    )
    if degree >= 2 and colors_rest.shape[1] >= 8:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        rgb = (
            rgb
            + 1.0925484305920792 * xy[:, None] * colors_rest[:, 3]
            + -1.0925484305920792 * yz[:, None] * colors_rest[:, 4]
            + 0.31539156525252005 * (2 * zz - xx - yy)[:, None] * colors_rest[:, 5]
            + -1.0925484305920792 * xz[:, None] * colors_rest[:, 6]
            + 0.5462742152960396 * (xx - yy)[:, None] * colors_rest[:, 7]
        )
    return (0.5 + rgb).clamp(0, 1)


def _accumulate_loop(
    acc_rgb: torch.Tensor,
    acc_a: torch.Tensor,
    m2: torch.Tensor,
    inv00: torch.Tensor,
    inv01: torch.Tensor,
    inv11: torch.Tensor,
    rad: torch.Tensor,
    opac: torch.Tensor,
    depth_w: torch.Tensor,
    rgbs: torch.Tensor,
    width: int,
    height: int,
) -> None:
    """Legacy per-Gaussian Python loop (slow reference)."""
    device, dtype = m2.device, m2.dtype
    W = width
    for i in range(m2.shape[0]):
        r = int(rad[i].detach().item())
        if r < 1:
            continue
        cx = m2[i, 0]
        cy = m2[i, 1]
        cx_d = float(cx.detach())
        cy_d = float(cy.detach())
        x0 = max(0, int(math.floor(cx_d - r)))
        x1 = min(width, int(math.ceil(cx_d + r)) + 1)
        y0 = max(0, int(math.floor(cy_d - r)))
        y1 = min(height, int(math.ceil(cy_d + r)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        ys = torch.arange(y0, y1, device=device, dtype=dtype)
        xs = torch.arange(x0, x1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dx = xx - cx
        dy = yy - cy
        power = -0.5 * (
            inv00[i] * dx * dx + 2.0 * inv01[i] * dx * dy + inv11[i] * dy * dy
        )
        alpha = (opac[i] * torch.exp(power.clamp(min=-40.0)) * depth_w[i]).clamp(
            max=0.99
        )
        flat = (yy.long() * W + xx.long()).reshape(-1)
        w = alpha.reshape(-1)
        acc_a.index_add_(0, flat, w)
        acc_rgb.index_add_(0, flat, w.unsqueeze(-1) * rgbs[i].unsqueeze(0))


def _accumulate_batched(
    acc_rgb: torch.Tensor,
    acc_a: torch.Tensor,
    m2: torch.Tensor,
    inv00: torch.Tensor,
    inv01: torch.Tensor,
    inv11: torch.Tensor,
    rad: torch.Tensor,
    opac: torch.Tensor,
    depth_w: torch.Tensor,
    rgbs: torch.Tensor,
    width: int,
    height: int,
    batch_size: int,
) -> None:
    """Vectorized stencil scatter: many Gaussians per index_add."""
    device, dtype = m2.device, m2.dtype
    n = m2.shape[0]
    W, H = width, height
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        cx = m2[sl, 0]
        cy = m2[sl, 1]
        r = rad[sl]
        r_max = int(r.detach().max().item()) if r.numel() else 0
        if r_max < 1:
            continue
        offs = torch.arange(-r_max, r_max + 1, device=device)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        ox = ox.reshape(-1)
        oy = oy.reshape(-1)
        # Integer pixel grid anchored at floor(center)
        base_x = torch.floor(cx.detach()).long()
        base_y = torch.floor(cy.detach()).long()
        pix_x = base_x[:, None] + ox[None, :]
        pix_y = base_y[:, None] + oy[None, :]
        dx = pix_x.to(dtype) - cx[:, None]
        dy = pix_y.to(dtype) - cy[:, None]
        r_lim = r[:, None] + 0.5
        inside = (
            (pix_x >= 0)
            & (pix_x < W)
            & (pix_y >= 0)
            & (pix_y < H)
            & (dx * dx + dy * dy <= r_lim * r_lim)
            & (r[:, None] >= 1)
        )
        if not bool(inside.any()):
            continue
        power = -0.5 * (
            inv00[sl, None] * dx * dx
            + 2.0 * inv01[sl, None] * dx * dy
            + inv11[sl, None] * dy * dy
        )
        alpha = (
            opac[sl, None]
            * torch.exp(power.clamp(min=-40.0))
            * depth_w[sl, None]
        ).clamp(max=0.99)
        alpha = torch.where(inside, alpha, alpha.new_zeros(()))
        flat = (pix_y * W + pix_x).reshape(-1)
        w = alpha.reshape(-1)
        # Drop empty slots (outside) to keep index_add smaller
        mask = inside.reshape(-1)
        if not bool(mask.any()):
            continue
        flat_m = flat[mask]
        w_m = w[mask]
        # rgbs: (b,3) -> (b,K,3) -> masked
        b = cx.shape[0]
        k = ox.numel()
        rgb_exp = rgbs[sl].unsqueeze(1).expand(b, k, 3).reshape(-1, 3)[mask]
        acc_a.index_add_(0, flat_m, w_m)
        acc_rgb.index_add_(0, flat_m, w_m.unsqueeze(-1) * rgb_exp)


def _tile_ranges(
    m2: torch.Tensor,
    rad: torch.Tensor,
    width: int,
    height: int,
    tile: int,
) -> list[torch.Tensor]:
    """Gaussian indices overlapping each image tile (for cache-friendly order)."""
    if tile <= 0 or m2.numel() == 0:
        return [torch.arange(m2.shape[0], device=m2.device, dtype=torch.long)]
    n_tx = (width + tile - 1) // tile
    n_ty = (height + tile - 1) // tile
    cx = m2[:, 0].detach()
    cy = m2[:, 1].detach()
    r = rad.detach()
    buckets: list[list[int]] = [[] for _ in range(n_tx * n_ty)]
    for i in range(m2.shape[0]):
        ri = float(r[i])
        x0 = max(0, int(math.floor(float(cx[i]) - ri)))
        x1 = min(width - 1, int(math.ceil(float(cx[i]) + ri)))
        y0 = max(0, int(math.floor(float(cy[i]) - ri)))
        y1 = min(height - 1, int(math.ceil(float(cy[i]) + ri)))
        tx0, tx1 = x0 // tile, x1 // tile
        ty0, ty1 = y0 // tile, y1 // tile
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                buckets[ty * n_tx + tx].append(i)
    out: list[torch.Tensor] = []
    for b in buckets:
        if b:
            out.append(torch.tensor(b, device=m2.device, dtype=torch.long))
    return out if out else [torch.arange(m2.shape[0], device=m2.device, dtype=torch.long)]


def rasterize_gaussians_cpu(
    means: torch.Tensor,
    scales_log: torch.Tensor,
    quats: torch.Tensor,
    opacities_logit: torch.Tensor,
    colors_dc: torch.Tensor,
    colors_rest: Optional[torch.Tensor],
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int = 1,
    near: float = 0.01,
    max_gaussians: int = 25000,
    max_radius: int = 30,
    depth_gamma: float = 4.0,
    batch_size: int = 64,
    tile_size: int = 0,
    method: str = "batched",
    num_threads: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return rgb (H,W,3), alpha (H,W). Depth-weighted soft composite (differentiable).

    method:
      batched - stencil batches (default, fast); tile_size>0 groups by image tiles
      loop    - legacy per-Gaussian Python loop
    """
    if num_threads is None:
        num_threads = max(1, min(32, os.cpu_count() or 4))
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(int(num_threads))

    try:
        means2d, cov2d, depths, radii, valid = project_gaussians(
            means, scales_log, quats, viewmat, K, width, height, near=near
        )
        idx = valid.nonzero(as_tuple=False).squeeze(-1)
        device, dtype = means.device, means.dtype
        acc_rgb = torch.zeros(height * width, 3, device=device, dtype=dtype)
        acc_a = torch.zeros(height * width, device=device, dtype=dtype)
        if idx.numel() == 0:
            z = means.new_zeros(height, width, 3)
            return z, means.new_zeros(height, width)

        order = torch.argsort(depths[idx])
        idx = idx[order]
        if idx.numel() > max_gaussians:
            idx = idx[:max_gaussians]

        Rcw = viewmat[:3, :3]
        tcw = viewmat[:3, 3]
        means_c = (Rcw @ means[idx].T).T + tcw
        dirs = F.normalize(-means_c, dim=-1)
        rest = colors_rest[idx] if colors_rest is not None else None
        rgbs = eval_sh(colors_dc[idx], rest, dirs, sh_degree)
        opac = torch.sigmoid(opacities_logit[idx]).clamp(max=0.99)

        m2 = means2d[idx]
        c2 = cov2d[idx]
        rad = radii[idx].clamp(max=float(max_radius))
        d = depths[idx]
        d_min = d.min()
        d_max = d.max().clamp_min(d_min + 1e-3)
        depth_w = torch.exp(-depth_gamma * (d - d_min) / (d_max - d_min + 1e-6))

        a = c2[:, 0, 0]
        b = c2[:, 0, 1]
        c = c2[:, 1, 1]
        det = (a * c - b * b).clamp_min(1e-8)
        inv00, inv01, inv11 = c / det, -b / det, a / det

        if method == "loop":
            _accumulate_loop(
                acc_rgb, acc_a, m2, inv00, inv01, inv11, rad, opac, depth_w, rgbs,
                width, height,
            )
        else:
            groups = _tile_ranges(m2, rad, width, height, tile_size)
            for gidx in groups:
                _accumulate_batched(
                    acc_rgb,
                    acc_a,
                    m2[gidx],
                    inv00[gidx],
                    inv01[gidx],
                    inv11[gidx],
                    rad[gidx],
                    opac[gidx],
                    depth_w[gidx],
                    rgbs[gidx],
                    width,
                    height,
                    batch_size=batch_size,
                )

        denom = acc_a.clamp_min(1e-6).unsqueeze(-1)
        rgb = (acc_rgb / denom).view(height, width, 3)
        alph = (1.0 - torch.exp(-acc_a)).view(height, width)
        return rgb, alph
    finally:
        torch.set_num_threads(prev_threads)
