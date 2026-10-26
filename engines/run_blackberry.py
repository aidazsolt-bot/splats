#!/usr/bin/env python3
"""End-to-end deterministic splat + mesh from a COLMAP+RGBA dataset (CPU, no training)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geom.dense_cpu import build_dense_cloud, cloud_to_open3d  # noqa: E402
from mesh.build_mesh import build_poisson_mesh  # noqa: E402
from prep.dataset import load_dataset, save_scaled_colmap  # noqa: E402
from splat.build_gaussians import build_deterministic_splat  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deterministic Gaussian splat + Poisson mesh (CPU, no training)"
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("/usr/src/TestAdnroidNDKSDK/blackberry-splats") / "dataset",
        help="Dataset root (cameras.txt, images.txt, points3D.txt, images/)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/usr/src/TestAdnroidNDKSDK/blackberry-splats") / "engines_out",
        help="Output directory",
    )
    ap.add_argument("--downscale", type=int, default=4, help="Image downscale for projection/bake")
    ap.add_argument("--densify-rounds", type=int, default=1)
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--stereo", action="store_true", help="Add ORB stereo triangulation points")
    ap.add_argument("--poisson-depth", type=int, default=9)
    ap.add_argument("--max-gaussians", type=int, default=120000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-views", type=int, default=None, help="Optional cap on number of views")
    ap.add_argument("--skip-mesh", action="store_true")
    ap.add_argument("--skip-splat", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[1/5] Loading dataset from {args.data}")
    ds = load_dataset(args.data, scale_intrinsics=True, max_views=args.max_views)
    prep_dir = out / "colmap_scaled"
    save_scaled_colmap(ds, prep_dir)
    cam = next(iter(ds.cameras.values()))
    print(
        f"  views={len(ds.views)} points3D={len(ds.points)} "
        f"image={ds.image_wh[0]}x{ds.image_wh[1]} cam_model={cam.model} "
        f"fx={cam.params[0]:.2f}"
    )

    print("[2/5] Building dense cloud (mask filter + kNN densify)")
    cloud = build_dense_cloud(
        ds,
        downscale=args.downscale,
        min_views=args.min_views,
        densify_rounds=args.densify_rounds,
        stereo=args.stereo,
        workers=args.workers,
    )
    print(f"  cloud points={len(cloud.xyz)}")
    # Save intermediate point cloud
    try:
        import open3d as o3d

        pcd = cloud_to_open3d(cloud)
        o3d.io.write_point_cloud(str(out / "dense_cloud.ply"), pcd)
    except Exception as e:
        print(f"  warn: could not write dense_cloud.ply: {e}")

    splat_path = out / "deterministic_splat.ply"
    if not args.skip_splat:
        print("[3/5] Building deterministic Gaussians")
        build_deterministic_splat(
            ds,
            cloud,
            splat_path,
            downscale=args.downscale,
            max_gaussians=args.max_gaussians,
            workers=args.workers,
        )
        print(f"  wrote {splat_path} ({splat_path.stat().st_size} bytes)")
    else:
        print("[3/5] Skipping splat")

    mesh_path = out / "mesh.obj"
    if not args.skip_mesh:
        print("[4/5] Poisson mesh + vertex color bake")
        try:
            obj, preview = build_poisson_mesh(
                cloud,
                ds,
                mesh_path,
                depth=args.poisson_depth,
                downscale=args.downscale,
                workers=args.workers,
            )
            print(f"  wrote {obj}")
            if preview:
                print(f"  preview {preview}")
        except Exception as e:
            print(f"  mesh failed: {e}")
    else:
        print("[4/5] Skipping mesh")

    # Optional copy of Danny reference splat if present next to zip extract
    print("[5/5] Summary")
    ref = Path("/usr/src/TestAdnroidNDKSDK/blackberry-splats") / "extract" / "output" / "splat_45000.ply"
    if ref.is_file():
        dest = out / "danny_splat_45000.ply"
        if not dest.exists():
            shutil.copy2(ref, dest)
            print(f"  copied reference {dest}")

    meta = {
        "data": str(args.data),
        "out": str(out),
        "n_views": len(ds.views),
        "n_sparse_points": len(ds.points),
        "n_cloud_points": int(len(cloud.xyz)),
        "image_wh": list(ds.image_wh),
        "downscale": args.downscale,
        "elapsed_sec": round(time.time() - t0, 2),
        "splat": str(splat_path) if splat_path.exists() else None,
        "mesh": str(mesh_path) if mesh_path.exists() else None,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Done in {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
