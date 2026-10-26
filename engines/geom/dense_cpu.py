"""CPU geometry densification from COLMAP sparse points + alpha masks.

Strategy (deterministic, no learning):
1. Start from COLMAP points3D (already multi-view triangulated).
2. Keep points that project into enough alpha-masked views.
3. Densify by mid-edge insertion of kNN edges (subdivision in 3D) and
   optional local jitter along estimated normals - still closed-form.
4. Optionally run a light OpenCV stereo pass on view pairs for extra depths
   (disabled by default for speed; enable with stereo=True).
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

# Allow running as script or package
_ENGINES = Path(__file__).resolve().parents[1]
if str(_ENGINES) not in sys.path:
    sys.path.insert(0, str(_ENGINES))

from prep.dataset import Dataset, ViewData, load_rgba, scaled_K  # noqa: E402


@dataclass
class DenseCloud:
    xyz: np.ndarray  # (N, 3) float64
    rgb: np.ndarray  # (N, 3) uint8
    normals: np.ndarray  # (N, 3) float64
    weights: np.ndarray  # (N,) visibility / confidence


def _project_points(xyz: np.ndarray, view: ViewData, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to pixels. Returns u, v, depth (z in camera)."""
    Xc = (view.R @ xyz.T).T + view.tvec  # (N, 3)
    z = Xc[:, 2]
    # Avoid div0
    z_safe = np.where(np.abs(z) < 1e-8, 1e-8, z)
    x = Xc[:, 0] / z_safe
    y = Xc[:, 1] / z_safe
    # SIMPLE_RADIAL distortion approx on normalized coords
    if view.dist is not None and float(np.abs(view.dist[0])) > 1e-12:
        k1 = float(view.dist[0])
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2
        x = x * radial
        y = y * radial
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return u, v, z


def _sample_alpha(alpha: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = alpha.shape
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    out = np.zeros(u.shape[0], dtype=np.float32)
    if np.any(inside):
        out[inside] = alpha[vi[inside], ui[inside]]
    return out


def _sample_rgb(rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    out = np.zeros((u.shape[0], 3), dtype=np.float32)
    if np.any(inside):
        out[inside] = rgb[vi[inside], ui[inside]].astype(np.float32)
    return out


def filter_points_by_mask(
    dataset: Dataset,
    downscale: int = 4,
    min_views: int = 2,
    alpha_thresh: float = 0.15,
    workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return xyz, rgb, visibility_count for sparse points surviving alpha."""
    pts = list(dataset.points.values())
    if not pts:
        raise RuntimeError("No points3D in dataset")
    xyz = np.stack([p.xyz for p in pts], axis=0)
    rgb = np.stack([p.rgb for p in pts], axis=0).astype(np.uint8)
    n = xyz.shape[0]
    vis = np.zeros(n, dtype=np.int32)

    def process_view(view: ViewData) -> np.ndarray:
        rgb_img, alpha = load_rgba(view.path, downscale=downscale)
        K = scaled_K(view.K, downscale)
        u, v, z = _project_points(xyz, view, K)
        front = z > 0
        a = _sample_alpha(alpha, u, v)
        return (front & (a >= alpha_thresh)).astype(np.int32)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(process_view, v) for v in dataset.views]
        for fut in as_completed(futs):
            vis += fut.result()

    keep = vis >= min_views
    return xyz[keep], rgb[keep], vis[keep].astype(np.float64)


def estimate_normals(xyz: np.ndarray, k: int = 16) -> np.ndarray:
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=min(k + 1, len(xyz)))
    normals = np.zeros_like(xyz)
    for i in range(len(xyz)):
        nb = xyz[idx[i, 1:]] if idx.ndim == 2 else xyz[[idx[i]]]
        if nb.shape[0] < 3:
            normals[i] = np.array([0.0, 0.0, 1.0])
            continue
        cov = np.cov(nb.T)
        w, v = np.linalg.eigh(cov)
        n = v[:, 0]
        normals[i] = n / (np.linalg.norm(n) + 1e-12)
    # Orient toward centroid of camera centers if available externally - leave as is
    return normals


def densify_knn_midpoints(
    xyz: np.ndarray,
    rgb: np.ndarray,
    weights: np.ndarray,
    knn: int = 8,
    max_edge: Optional[float] = None,
    rounds: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert midpoints on short kNN edges (deterministic subdivision)."""
    cur_xyz, cur_rgb, cur_w = xyz, rgb, weights
    for _ in range(rounds):
        tree = cKDTree(cur_xyz)
        k = min(knn + 1, len(cur_xyz))
        dists, idx = tree.query(cur_xyz, k=k)
        if max_edge is None:
            # Adaptive: median of nearest-neighbor distances * 1.5
            nn = dists[:, 1] if dists.ndim == 2 else dists
            max_edge_r = float(np.median(nn) * 1.75)
        else:
            max_edge_r = max_edge

        new_pts = []
        new_cols = []
        new_ws = []
        seen = set()
        for i in range(len(cur_xyz)):
            for j_pos in range(1, k):
                j = int(idx[i, j_pos])
                d = float(dists[i, j_pos])
                if d <= 0 or d > max_edge_r:
                    continue
                a, b = (i, j) if i < j else (j, i)
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                mid = 0.5 * (cur_xyz[i] + cur_xyz[j])
                col = (0.5 * (cur_rgb[i].astype(np.float32) + cur_rgb[j].astype(np.float32))).astype(np.uint8)
                w = 0.5 * (cur_w[i] + cur_w[j])
                new_pts.append(mid)
                new_cols.append(col)
                new_ws.append(w)
        if not new_pts:
            break
        cur_xyz = np.vstack([cur_xyz, np.stack(new_pts, axis=0)])
        cur_rgb = np.vstack([cur_rgb, np.stack(new_cols, axis=0)])
        cur_w = np.concatenate([cur_w, np.asarray(new_ws, dtype=np.float64)])
    return cur_xyz, cur_rgb, cur_w


def stereo_pair_points(
    view_a: ViewData,
    view_b: ViewData,
    downscale: int = 8,
    max_points: int = 5000,
    alpha_thresh: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse stereo via ORB matches + triangulation (deterministic OpenCV)."""
    rgb_a, alpha_a = load_rgba(view_a.path, downscale=downscale)
    rgb_b, alpha_b = load_rgba(view_b.path, downscale=downscale)
    gray_a = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2GRAY)
    # Mask keypoints to subject
    mask_a = (alpha_a >= alpha_thresh).astype(np.uint8) * 255
    mask_b = (alpha_b >= alpha_thresh).astype(np.uint8) * 255

    orb = cv2.ORB_create(nfeatures=4000)
    kp_a, des_a = orb.detectAndCompute(gray_a, mask_a)
    kp_b, des_b = orb.detectAndCompute(gray_b, mask_b)
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_b)
    matches = sorted(matches, key=lambda m: m.distance)[: max_points * 2]
    if len(matches) < 8:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    pts_a = np.array([kp_a[m.queryIdx].pt for m in matches], dtype=np.float64)
    pts_b = np.array([kp_b[m.trainIdx].pt for m in matches], dtype=np.float64)
    Ka = scaled_K(view_a.K, downscale)
    Kb = scaled_K(view_b.K, downscale)

    # Projection matrices P = K [R|t]
    Pa = Ka @ np.hstack([view_a.R, view_a.tvec.reshape(3, 1)])
    Pb = Kb @ np.hstack([view_b.R, view_b.tvec.reshape(3, 1)])
    X_h = cv2.triangulatePoints(Pa, Pb, pts_a.T, pts_b.T)
    X = (X_h[:3] / np.where(np.abs(X_h[3]) < 1e-12, 1e-12, X_h[3])).T

    # Keep points in front of both cameras and inside masks
    ua, va, za = _project_points(X, view_a, Ka)
    ub, vb, zb = _project_points(X, view_b, Kb)
    aa = _sample_alpha(alpha_a, ua, va)
    ab = _sample_alpha(alpha_b, ub, vb)
    ok = (za > 0) & (zb > 0) & (aa >= alpha_thresh) & (ab >= alpha_thresh)
    X = X[ok]
    cols = _sample_rgb(rgb_a, ua, va)[ok].astype(np.uint8)
    if X.shape[0] > max_points:
        sel = np.linspace(0, X.shape[0] - 1, max_points).astype(np.int64)
        X = X[sel]
        cols = cols[sel]
    return X, cols


def build_dense_cloud(
    dataset: Dataset,
    downscale: int = 4,
    min_views: int = 2,
    densify_rounds: int = 1,
    knn: int = 8,
    stereo: bool = False,
    stereo_pairs: int = 12,
    workers: int = 8,
    orient_normals_to_cameras: bool = True,
) -> DenseCloud:
    xyz, rgb, weights = filter_points_by_mask(
        dataset, downscale=downscale, min_views=min_views, workers=workers
    )
    xyz, rgb, weights = densify_knn_midpoints(
        xyz, rgb, weights, knn=knn, rounds=densify_rounds
    )

    if stereo and len(dataset.views) >= 2:
        # Pick pairs with moderate baseline (neighboring in image_id order)
        views = dataset.views
        step = max(1, len(views) // max(1, stereo_pairs))
        extra_xyz = []
        extra_rgb = []
        for i in range(0, len(views) - 1, step):
            j = min(len(views) - 1, i + max(1, step // 2))
            if i == j:
                continue
            X, C = stereo_pair_points(views[i], views[j], downscale=max(downscale, 6))
            if X.shape[0]:
                extra_xyz.append(X)
                extra_rgb.append(C)
        if extra_xyz:
            xyz = np.vstack([xyz, *extra_xyz])
            rgb = np.vstack([rgb, *extra_rgb])
            weights = np.concatenate(
                [weights, np.full(sum(x.shape[0] for x in extra_xyz), 2.0, dtype=np.float64)]
            )

    normals = estimate_normals(xyz, k=16)
    if orient_normals_to_cameras:
        cam_c = np.mean(np.stack([v.C for v in dataset.views], axis=0), axis=0)
        to_cam = cam_c.reshape(1, 3) - xyz
        flip = np.sum(normals * to_cam, axis=1) < 0
        normals[flip] *= -1.0

    return DenseCloud(xyz=xyz, rgb=rgb, normals=normals, weights=weights)


def cloud_to_open3d(cloud: DenseCloud):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cloud.rgb.astype(np.float64) / 255.0)
    pcd.normals = o3d.utility.Vector3dVector(cloud.normals.astype(np.float64))
    return pcd
