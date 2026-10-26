"""Quality-first photometric losses (CPU)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_masked(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask[..., None]
    denom = (m.sum() * pred.shape[-1]).clamp_min(1.0)
    return ((pred - gt).abs() * m).sum() / denom


def ssim_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """pred/gt H W C -> 1-SSIM."""
    pred_n = pred.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    gt_n = gt.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    if mask is not None:
        m = mask[None, None]
        pred_n = pred_n * m
        gt_n = gt_n * m
    return 1.0 - _ssim(pred_n, gt_n)


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    C1, C2 = 0.01**2, 0.03**2
    pad = window_size // 2
    channel = img1.size(1)
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - pad
    g = torch.exp(-(coords**2) / 2.0)
    g = g / g.sum()
    w1 = g.unsqueeze(1)
    window = (w1 @ w1.t()).unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size)
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    s2 = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    s12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean()


def alpha_bce(pred_a: torch.Tensor, gt_a: torch.Tensor) -> torch.Tensor:
    p = pred_a.clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(p, gt_a.clamp(0, 1))


class OptionalLPIPS:
    """Lazy LPIPS (Alex) on CPU if package installed."""

    def __init__(self) -> None:
        self._fn = None
        try:
            import lpips

            self._fn = lpips.LPIPS(net="alex").eval()
            for p in self._fn.parameters():
                p.requires_grad_(False)
        except Exception:
            self._fn = None

    @property
    def available(self) -> bool:
        return self._fn is not None

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self._fn is None:
            return pred.new_zeros(())
        # pred/gt H W C -> NCHW in [-1,1]
        device = next(self._fn.parameters()).device
        p = pred.permute(2, 0, 1).unsqueeze(0).to(device)
        g = gt.permute(2, 0, 1).unsqueeze(0).to(device)
        m = mask[None, None].to(device)
        p = (p * m) * 2 - 1
        g = (g * m) * 2 - 1
        return self._fn(p, g).mean()
