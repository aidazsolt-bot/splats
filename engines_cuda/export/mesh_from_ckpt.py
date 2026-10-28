"""Mesh from optimized Gaussian means (post-train).

Backends:
  open3d     - Poisson reconstruction (default, usually better surfaces)
  pytorch3d  - voxel occupancy + cubify (needs optional pytorch3d install)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_means_rgb(ckpt_path: Path, opacity_min: float) -> tuple[np.ndarray, np.ndarray]:
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    means = ckpt["means"].numpy()
    opac = torch.sigmoid(ckpt["opacities"]).numpy().reshape(-1)
    dc = ckpt["colors_dc"].numpy()
    c0 = 0.28209479177387814
    rgb = np.clip(0.5 + c0 * dc, 0.0, 1.0)
    keep = opac >= opacity_min
    return means[keep].astype(np.float64), rgb[keep].astype(np.float64)


def _colorize_vertices(
    verts: np.ndarray, pts: np.ndarray, rgb: np.ndarray
) -> np.ndarray:
    from scipy.spatial import cKDTree

    _, idx = cKDTree(pts).query(verts, k=1)
    return rgb[idx]


def mesh_open3d_poisson(
    means: np.ndarray,
    rgb: np.ndarray,
    out: Path,
    poisson_depth: int,
) -> None:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )
    dens = np.asarray(densities)
    if dens.size:
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.02))
    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        _colorize_vertices(np.asarray(mesh.vertices), means, rgb)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out), mesh, write_vertex_colors=True)
    print(
        f"Wrote {out} backend=open3d verts={len(mesh.vertices)} "
        f"tris={len(mesh.triangles)}",
        flush=True,
    )


def mesh_pytorch3d_cubify(
    means: np.ndarray,
    rgb: np.ndarray,
    out: Path,
    grid_size: int,
    thresh: float,
) -> None:
    try:
        from pytorch3d.io import IO
        from pytorch3d.ops import cubify
        from pytorch3d.renderer import TexturesVertex
        from pytorch3d.structures import Meshes
    except ImportError as e:
        raise SystemExit(
            "pytorch3d is required for --backend pytorch3d. "
            "Install a wheel matching your torch/CUDA, e.g. Colab: "
            'pip install "git+https://github.com/facebookresearch/pytorch3d.git" '
            f"Original error: {e}"
        ) from e

    if means.shape[0] < 8:
        raise SystemExit("Too few points after opacity filter for cubify mesh")

    lo = means.min(axis=0)
    hi = means.max(axis=0)
    extent = np.maximum(hi - lo, 1e-6)
    pad = 0.05 * extent
    lo = lo - pad
    hi = hi + pad
    extent = hi - lo

    # Map world -> [0, grid_size-1]
    g = float(grid_size - 1)
    ijk = np.floor((means - lo) / extent * g).astype(np.int64)
    ijk = np.clip(ijk, 0, grid_size - 1)
    voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    # cubify expects (N, D, H, W) with D along z; use z,y,x indexing
    voxels[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = 1.0

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vox_t = torch.from_numpy(voxels).unsqueeze(0).to(device)
    meshes = cubify(vox_t, thresh=thresh)
    verts = meshes.verts_packed().detach().cpu().numpy()
    faces = meshes.faces_packed().detach().cpu().numpy()
    if verts.size == 0 or faces.size == 0:
        raise SystemExit(
            "pytorch3d cubify produced an empty mesh; try lower --voxel-thresh "
            "or higher --grid-size"
        )

    # cubify verts are in [0, D] style voxel coords -> world
    verts_world = lo + (verts / g) * extent
    colors = _colorize_vertices(verts_world, means, rgb)

    out.parent.mkdir(parents=True, exist_ok=True)
    verts_t = torch.from_numpy(verts_world.astype(np.float32))
    faces_t = torch.from_numpy(faces.astype(np.int64))
    cols_t = torch.from_numpy(colors.astype(np.float32))
    mesh_p3d = Meshes(
        verts=[verts_t],
        faces=[faces_t],
        textures=TexturesVertex(verts_features=[cols_t]),
    )
    IO().save_mesh(mesh_p3d, str(out))
    print(
        f"Wrote {out} backend=pytorch3d verts={verts_world.shape[0]} "
        f"tris={faces.shape[0]} grid={grid_size}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mesh from trained Gaussian means (Open3D Poisson or PyTorch3D cubify)"
    )
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--backend",
        choices=("open3d", "pytorch3d"),
        default="open3d",
        help="open3d=Poisson (default); pytorch3d=voxel cubify",
    )
    ap.add_argument("--opacity-min", type=float, default=0.05)
    ap.add_argument("--poisson-depth", type=int, default=9)
    ap.add_argument(
        "--grid-size",
        type=int,
        default=128,
        help="pytorch3d only: occupancy grid resolution per axis",
    )
    ap.add_argument(
        "--voxel-thresh",
        type=float,
        default=0.5,
        help="pytorch3d only: cubify occupancy threshold",
    )
    args = ap.parse_args()

    means, rgb = _load_means_rgb(args.ckpt, args.opacity_min)
    print(f"Points after opacity filter: {means.shape[0]}", flush=True)

    if args.backend == "open3d":
        mesh_open3d_poisson(means, rgb, args.out, args.poisson_depth)
    else:
        mesh_pytorch3d_cubify(
            means, rgb, args.out, args.grid_size, args.voxel_thresh
        )


if __name__ == "__main__":
    main()
