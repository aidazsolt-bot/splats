"""Build a Poisson mesh from optimized Gaussian means (post-train, CPU Open3D)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description="Poisson mesh from trained Gaussian means")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opacity-min", type=float, default=0.05)
    ap.add_argument("--poisson-depth", type=int, default=9)
    args = ap.parse_args()

    import open3d as o3d

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    means = ckpt["means"].numpy()
    opac = torch.sigmoid(ckpt["opacities"]).numpy().reshape(-1)
    dc = ckpt["colors_dc"].numpy()
    # SH DC -> RGB
    c0 = 0.28209479177387814
    rgb = np.clip(0.5 + c0 * dc, 0.0, 1.0)
    keep = opac >= args.opacity_min
    means = means[keep]
    rgb = rgb[keep]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.poisson_depth
    )
    dens = np.asarray(densities)
    if dens.size:
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.02))
    mesh.compute_vertex_normals()
    # transfer colors via nearest neighbor
    tree_pts = np.asarray(pcd.points)
    from scipy.spatial import cKDTree

    _, idx = cKDTree(tree_pts).query(np.asarray(mesh.vertices), k=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgb[idx])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.out), mesh, write_vertex_colors=True)
    print(f"Wrote {args.out} verts={len(mesh.vertices)} tris={len(mesh.triangles)}")


if __name__ == "__main__":
    main()
