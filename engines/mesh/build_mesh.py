"""Poisson surface reconstruction + multi-view vertex color bake (CPU, no training)."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_ENGINES = Path(__file__).resolve().parents[1]
if str(_ENGINES) not in sys.path:
    sys.path.insert(0, str(_ENGINES))

from geom.dense_cpu import DenseCloud, cloud_to_open3d, _project_points, _sample_alpha, _sample_rgb  # noqa: E402
from prep.dataset import Dataset, load_rgba, scaled_K  # noqa: E402


def bake_vertex_colors(
    dataset: Dataset,
    vertices: np.ndarray,
    normals: np.ndarray,
    downscale: int = 4,
    alpha_thresh: float = 0.1,
    workers: int = 8,
) -> np.ndarray:
    """Weighted multi-view average color for mesh vertices. Returns float RGB in [0,1]."""
    n = vertices.shape[0]
    acc = np.zeros((n, 3), dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)

    def process(view):
        rgb_img, alpha = load_rgba(view.path, downscale=downscale)
        K = scaled_K(view.K, downscale)
        u, v, z = _project_points(vertices, view, K)
        front = z > 0
        to_cam = view.C.reshape(1, 3) - vertices
        to_cam_n = to_cam / (np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-12)
        # Prefer views where normal faces camera
        ndot = np.clip(np.sum(normals * to_cam_n, axis=1), 0.0, 1.0)
        a = _sample_alpha(alpha, u, v)
        cols = _sample_rgb(rgb_img, u, v)
        w = front.astype(np.float64) * a.astype(np.float64) * (ndot ** 2)
        w = np.where(a >= alpha_thresh, w, 0.0)
        return w, cols

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(process, v) for v in dataset.views]
        for fut in as_completed(futs):
            w, cols = fut.result()
            acc += cols * w.reshape(-1, 1)
            wsum += w

    rgb = acc / np.maximum(wsum, 1e-8).reshape(-1, 1) / 255.0
    missing = wsum < 1e-6
    rgb[missing] = 0.5
    return np.clip(rgb, 0.0, 1.0)


def build_poisson_mesh(
    cloud: DenseCloud,
    dataset: Dataset,
    out_obj: Path,
    depth: int = 9,
    densify_scale: float = 1.1,
    density_quantile: float = 0.02,
    downscale: int = 4,
    workers: int = 8,
    bake_texture_png: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """Create Poisson mesh, bake vertex colors, write OBJ (+ optional texture atlas skip: vertex colors)."""
    import open3d as o3d

    pcd = cloud_to_open3d(cloud)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    cam_mean = np.mean(np.stack([v.C for v in dataset.views], axis=0), axis=0)
    pcd.orient_normals_towards_camera_location(camera_location=cam_mean)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=densify_scale, linear_fit=False
    )
    densities = np.asarray(densities)
    if densities.size and mesh.has_vertices():
        thresh = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < thresh)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    verts = np.asarray(mesh.vertices)
    if verts.shape[0] == 0:
        raise RuntimeError("Poisson mesh is empty - try lowering density_quantile or densify more")

    # Normals for bake
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    colors = bake_vertex_colors(
        dataset, verts, normals, downscale=downscale, workers=workers
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    out_obj = Path(out_obj)
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    # Also write PLY for easy viewing
    out_ply = out_obj.with_suffix(".ply")
    o3d.io.write_triangle_mesh(str(out_ply), mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(str(out_obj), mesh, write_vertex_colors=True)

    tex_path = None
    if bake_texture_png:
        # Simple solid bake of average colors as a debug strip (vertex colors are primary)
        tex_path = out_obj.with_name(out_obj.stem + "_vertex_preview.png")
        _write_color_preview(colors, tex_path)

    # MTL companion for viewers that expect it
    mtl = out_obj.with_suffix(".mtl")
    mtl.write_text(
        "# Vertex-colored mesh; no UV atlas\n"
        "newmtl material0\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        "Ks 0.000 0.000 0.000\n"
        "d 1.0\n"
        "illum 1\n",
        encoding="utf-8",
    )
    return out_obj, tex_path


def _write_color_preview(colors: np.ndarray, path: Path) -> None:
    from PIL import Image

    n = colors.shape[0]
    w = int(np.ceil(np.sqrt(n)))
    h = int(np.ceil(n / w))
    img = np.zeros((h * w, 3), dtype=np.uint8)
    img[:n] = (colors * 255).astype(np.uint8)
    img = img.reshape(h, w, 3)
    Image.fromarray(img, mode="RGB").save(path)
