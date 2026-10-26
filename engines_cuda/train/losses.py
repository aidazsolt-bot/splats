"""Photometric losses with optional alpha masking."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_masked(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/gt: ... H W C, mask: ... H W in {0,1}."""
    m = mask[..., None]
    denom = m.sum().clamp_min(1.0) * pred.shape[-1]
    return ((pred - gt).abs() * m).sum() / denom


def ssim_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Simple SSIM on NCHW; returns 1 - SSIM (to minimize). pred/gt: B H W C."""
    pred_n = pred.permute(0, 3, 1, 2).clamp(0, 1)
    gt_n = gt.permute(0, 3, 1, 2).clamp(0, 1)
    if mask is not None:
        # Soft-mask by multiplying (approximate; better than ignoring)
        m = mask[:, None]
        pred_n = pred_n * m
        gt_n = gt_n * m
    return 1.0 - _ssim(pred_n, gt_n)


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    """Mean SSIM over batch. img: B C H W."""
    pad = window_size // 2
    channel = img1.size(1)
    # Gaussian window
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - pad
    g = torch.exp(-(coords**2) / 2.0)
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = (window_1d @ window_1d.t()).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def alpha_bce(pred_alpha: torch.Tensor, gt_alpha: torch.Tensor) -> torch.Tensor:
    """pred_alpha/gt_alpha: B H W in [0,1]."""
    eps = 1e-6
    p = pred_alpha.clamp(eps, 1 - eps)
    g = gt_alpha.clamp(0, 1)
    return F.binary_cross_entropy(p, g)
