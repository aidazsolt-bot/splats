"""Mesh from optimized Gaussian means (post-train).

Backends:
  open3d     - Poisson reconstruction (default, usually better surfaces)
  pytorch3d  - voxel occupancy + cubify (needs pytorch3d)

Writes PLY plus Wavefront OBJ + MTL (vertex-color atlas as map_Kd).
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


def _write_color_atlas(colors: np.ndarray, png_path: Path) -> np.ndarray:
    """Pack per-vertex RGB into a PNG; return UVs in [0,1] (OpenGL-style, V up)."""
    from PIL import Image

    n = int(colors.shape[0])
    w = int(np.ceil(np.sqrt(max(n, 1))))
    h = int(np.ceil(n / w)) if n else 1
    img = np.zeros((h * w, 3), dtype=np.uint8)
    img[:n] = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(img.reshape(h, w, 3), mode="RGB").save(png_path)

    uvs = np.zeros((n, 2), dtype=np.float64)
    cols = np.arange(n) % w
    rows = np.arange(n) // w
    uvs[:, 0] = (cols + 0.5) / float(w)
    uvs[:, 1] = 1.0 - (rows + 0.5) / float(h)
    return uvs


def write_wavefront_obj_mtl(
    out_obj: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
) -> tuple[Path, Path, Path]:
    """Write OBJ + MTL + vertex-color atlas PNG. Faces are 0-based triangles."""
    out_obj = Path(out_obj)
    if out_obj.suffix.lower() != ".obj":
        out_obj = out_obj.with_suffix(".obj")
    out_obj.parent.mkdir(parents=True, exist_ok=True)

    stem = out_obj.stem
    mtl_name = f"{stem}.mtl"
    png_name = f"{stem}_atlas.png"
    out_mtl = out_obj.with_name(mtl_name)
    out_png = out_obj.with_name(png_name)

    uvs = _write_color_atlas(colors.astype(np.float64), out_png)
    mean_kd = np.clip(colors.mean(axis=0), 0.0, 1.0)

    out_mtl.write_text(
        "# Auto-generated from Gaussian means mesh export\n"
        "newmtl material0\n"
        "Ka 1.000000 1.000000 1.000000\n"
        f"Kd {mean_kd[0]:.6f} {mean_kd[1]:.6f} {mean_kd[2]:.6f}\n"
        "Ks 0.000000 0.000000 0.000000\n"
        "Ns 1.000000\n"
        "d 1.000000\n"
        "illum 1\n"
        f"map_Kd {png_name}\n",
        encoding="utf-8",
    )

    lines: list[str] = [
        "# Wavefront OBJ from engines_cuda/export/mesh_from_ckpt.py\n",
        f"mtllib {mtl_name}\n",
        "usemtl material0\n",
    ]
    for (x, y, z), (r, g, b) in zip(verts, colors):
        lines.append(
            f"v {x:.8f} {y:.8f} {z:.8f} {r:.6f} {g:.6f} {b:.6f}\n"
        )
    for u, vv in uvs:
        lines.append(f"vt {u:.8f} {vv:.8f}\n")
    for a, b, c in faces.astype(np.int64):
        # OBJ is 1-based; same index for v and vt
        i, j, k = a + 1, b + 1, c + 1
        lines.append(f"f {i}/{i} {j}/{j} {k}/{k}\n")
    out_obj.write_text("".join(lines), encoding="utf-8")
    print(
        f"Wrote {out_obj} + {out_mtl.name} + {out_png.name} "
        f"verts={verts.shape[0]} tris={faces.shape[0]}",
        flush=True,
    )
    return out_obj, out_mtl, out_png


def _save_mesh_outputs(
    out: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    backend: str,
) -> None:
    import open3d as o3d

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ply_path = out if out.suffix.lower() == ".ply" else out.with_suffix(".ply")
    obj_path = out if out.suffix.lower() == ".obj" else out.with_suffix(".obj")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(ply_path), mesh, write_vertex_colors=True)
    print(
        f"Wrote {ply_path} backend={backend} verts={verts.shape[0]} "
        f"tris={faces.shape[0]}",
        flush=True,
    )
    write_wavefront_obj_mtl(obj_path, verts, faces, colors)


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
    colors = _colorize_vertices(np.asarray(mesh.vertices), means, rgb)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    _save_mesh_outputs(
        out,
        np.asarray(mesh.vertices),
        np.asarray(mesh.triangles),
        colors,
        backend="open3d",
    )


def mesh_pytorch3d_cubify(
    means: np.ndarray,
    rgb: np.ndarray,
    out: Path,
    grid_size: int,
    thresh: float,
) -> None:
    try:
        from pytorch3d.ops import cubify
    except ImportError as e:
        raise SystemExit(
            "pytorch3d is required for --backend pytorch3d. "
            "Install with: python -m pip install --no-build-isolation "
            "-r engines_cuda/requirements-pytorch3d.txt "
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

    g = float(grid_size - 1)
    ijk = np.floor((means - lo) / extent * g).astype(np.int64)
    ijk = np.clip(ijk, 0, grid_size - 1)
    voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
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

    verts_world = lo + (verts / g) * extent
    colors = _colorize_vertices(verts_world, means, rgb)
    _save_mesh_outputs(out, verts_world, faces, colors, backend="pytorch3d")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Mesh from trained Gaussian means (Open3D Poisson or PyTorch3D cubify). "
            "Writes PLY and Wavefront OBJ+MTL with vertex-color atlas."
        )
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
