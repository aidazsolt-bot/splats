"""Build 3DGS-compatible PLY from a dense cloud without training.

For each point:
- covariance / scale / rotation from local PCA (eigendecomposition)
- SH DC color from weighted multi-view average (closed-form)
- opacity from visibility * alpha agreement formula
"""

from __future__ import annotations

import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

_ENGINES = Path(__file__).resolve().parents[1]
if str(_ENGINES) not in sys.path:
    sys.path.insert(0, str(_ENGINES))

from geom.dense_cpu import DenseCloud, _project_points, _sample_alpha, _sample_rgb  # noqa: E402
from prep.dataset import Dataset, load_rgba, scaled_K  # noqa: E402

# 3DGS SH DC encoding: color = 0.5 + C0 * sh_dc  with C0 = 0.28209479177387814
SH_C0 = 0.28209479177387814


def local_pca_scales_rots(
    xyz: np.ndarray,
    knn: int = 16,
    scale_percentile_cap: float = 99.0,
    min_scale: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return scales (N,3) linear scales, and quaternions (N,4) wxyz via local PCA."""
    n = len(xyz)
    tree = cKDTree(xyz)
    k = min(knn + 1, n)
    _, idx = tree.query(xyz, k=k)
    # idx: (N, k) neighbors including self at [:,0]
    nb = xyz[idx[:, 1:]]  # (N, k-1, 3)
    center = xyz[:, None, :]
    centered = nb - center
    # cov[i] = centered[i].T @ centered[i] / (k-2)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(1, k - 2)
    cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
    evals, evecs = np.linalg.eigh(cov)  # ascending
    evals = np.clip(evals, 0.0, None)
    # Reverse to largest-first
    evals = evals[:, ::-1]
    evecs = evecs[:, :, ::-1]
    scales = np.sqrt(evals + 1e-12)
    scales = np.maximum(scales, min_scale)

    # Fix left-handed frames
    dets = np.linalg.det(evecs)
    flip = dets < 0
    evecs[flip, :, 2] *= -1

    rots = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        rots[i] = _rotmat_to_quat(evecs[i])

    flat = scales.reshape(-1)
    cap = float(np.percentile(flat, scale_percentile_cap))
    scales = np.clip(scales, min_scale, max(cap, min_scale * 10))
    return scales, rots


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion (w, x, y, z)."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def multiview_color_and_opacity(
    dataset: Dataset,
    xyz: np.ndarray,
    downscale: int = 4,
    alpha_thresh: float = 0.1,
    workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form weighted average RGB and opacity in [0,1].

    Weight per view: alpha * cos^2(viewing angle) * front_mask.
    Opacity: sigmoid-like map of (mean_alpha * n_vis / n_views_cap).
    """
    n = xyz.shape[0]
    acc_rgb = np.zeros((n, 3), dtype=np.float64)
    acc_w = np.zeros(n, dtype=np.float64)
    acc_alpha = np.zeros(n, dtype=np.float64)
    acc_vis = np.zeros(n, dtype=np.float64)

    def process(view):
        rgb_img, alpha = load_rgba(view.path, downscale=downscale)
        K = scaled_K(view.K, downscale)
        u, v, z = _project_points(xyz, view, K)
        front = z > 0
        # Viewing cosine: direction from point to camera
        to_cam = view.C.reshape(1, 3) - xyz
        to_cam_n = to_cam / (np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-12)
        # Approximate surface normal from camera look (use -ray for weight)
        # Use angle between viewing direction and camera principal axis
        cam_fwd = view.R[2, :]  # camera +Z in world? R maps world->cam, so cam Z axis in world is R.T[:,2]
        cam_axis = view.R.T[:, 2]
        cosang = np.clip(np.sum((-to_cam_n) * cam_axis.reshape(1, 3), axis=1), 0.0, 1.0)
        a = _sample_alpha(alpha, u, v)
        cols = _sample_rgb(rgb_img, u, v)
        w = front.astype(np.float64) * (a.astype(np.float64)) * (cosang ** 2)
        w = np.where(a >= alpha_thresh, w, 0.0)
        return w, cols, a.astype(np.float64), (w > 0).astype(np.float64)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(process, v) for v in dataset.views]
        for fut in as_completed(futs):
            w, cols, a, vis = fut.result()
            acc_rgb += cols * w.reshape(-1, 1)
            acc_w += w
            acc_alpha += a * (w > 0)
            acc_vis += vis

    # Avoid div0: fallback to gray
    denom = np.maximum(acc_w, 1e-8)
    rgb = acc_rgb / denom.reshape(-1, 1)
    mean_a = acc_alpha / np.maximum(acc_vis, 1.0)
    # Opacity formula: 1 - exp(-beta * mean_alpha * log(1+vis))
    beta = 1.35
    opacity = 1.0 - np.exp(-beta * mean_a * np.log1p(acc_vis))
    opacity = np.clip(opacity, 0.05, 0.99)
    # Where never seen, keep low opacity and use cloud rgb later
    unseen = acc_vis < 1
    opacity[unseen] = 0.05
    return rgb, opacity


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0,1] -> f_dc (N,3)."""
    return ((rgb - 0.5) / SH_C0).astype(np.float32)


def opacity_to_logit(opacity: np.ndarray) -> np.ndarray:
    o = np.clip(opacity, 1e-4, 1.0 - 1e-4)
    return np.log(o / (1.0 - o)).astype(np.float32)


def scale_to_log(scales: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(scales, 1e-8)).astype(np.float32)


def write_gsplat_ply(
    path: Path,
    xyz: np.ndarray,
    normals: np.ndarray,
    f_dc: np.ndarray,
    opacity_logit: np.ndarray,
    scale_log: np.ndarray,
    rot_wxyz: np.ndarray,
    sh_rest_zeros: bool = True,
) -> None:
    """Write binary_little_endian PLY matching Postshot / 3DGS layout (SH degree 3)."""
    n = xyz.shape[0]
    # 45 rest SH coeffs (degree 3) set to 0
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(45):
        header.append(f"property float f_rest_{i}")
    header += [
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        # Pack each vertex: 3+3+3+45+1+3+4 = 62 floats
        for i in range(n):
            vals = [
                float(xyz[i, 0]),
                float(xyz[i, 1]),
                float(xyz[i, 2]),
                float(normals[i, 0]),
                float(normals[i, 1]),
                float(normals[i, 2]),
                float(f_dc[i, 0]),
                float(f_dc[i, 1]),
                float(f_dc[i, 2]),
            ]
            vals.extend([0.0] * 45)
            vals.append(float(opacity_logit[i]))
            vals.extend([float(scale_log[i, 0]), float(scale_log[i, 1]), float(scale_log[i, 2])])
            # rot stored as wxyz matching common 3DGS exporters
            vals.extend(
                [
                    float(rot_wxyz[i, 0]),
                    float(rot_wxyz[i, 1]),
                    float(rot_wxyz[i, 2]),
                    float(rot_wxyz[i, 3]),
                ]
            )
            f.write(struct.pack("<62f", *vals))


def build_deterministic_splat(
    dataset: Dataset,
    cloud: DenseCloud,
    out_ply: Path,
    downscale: int = 4,
    knn_pca: int = 16,
    max_gaussians: Optional[int] = None,
    workers: int = 8,
) -> Path:
    xyz = cloud.xyz
    normals = cloud.normals
    if max_gaussians is not None and len(xyz) > max_gaussians:
        # Deterministic subsample: stride
        step = int(np.ceil(len(xyz) / max_gaussians))
        sel = np.arange(0, len(xyz), step)[:max_gaussians]
        xyz = xyz[sel]
        normals = normals[sel]
        seed_rgb = cloud.rgb[sel].astype(np.float64) / 255.0
    else:
        seed_rgb = cloud.rgb.astype(np.float64) / 255.0

    scales, rots = local_pca_scales_rots(xyz, knn=knn_pca)
    mv_rgb, opacity = multiview_color_and_opacity(
        dataset, xyz, downscale=downscale, workers=workers
    )
    # mv_rgb is 0..255 float from image sampling
    rgb = np.clip(mv_rgb / 255.0, 0.0, 1.0)
    unseen = opacity <= 0.06
    rgb[unseen] = seed_rgb[unseen]

    f_dc = rgb_to_sh_dc(rgb)
    op_logit = opacity_to_logit(opacity)
    sc_log = scale_to_log(scales)
    write_gsplat_ply(out_ply, xyz, normals, f_dc, op_logit, sc_log, rots)
    return Path(out_ply)
