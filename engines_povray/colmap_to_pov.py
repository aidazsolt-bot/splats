"""Build a POV-Ray scene from COLMAP cameras + mesh/points + optional image textures."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_ENGINES = _REPO / "engines"
if str(_ENGINES) not in sys.path:
    sys.path.insert(0, str(_ENGINES))

from prep.colmap_io import (  # noqa: E402
    camera_K_dist,
    camera_center,
    qvec_to_rotmat,
    read_cameras_text,
    read_images_text,
    read_points3d_text,
    scale_camera_to_image_size,
)
from PIL import Image


def _probe_wh(images_dir: Path, name: str) -> Tuple[int, int]:
    with Image.open(images_dir / name) as im:
        return im.size


def fov_x_deg(fx: float, width: int) -> float:
    return float(2.0 * math.degrees(math.atan(0.5 * width / fx)))


def colmap_camera_to_pov(
    qvec: np.ndarray,
    tvec: np.ndarray,
    fx: float,
    width: int,
    height: int,
) -> dict:
    """COLMAP OpenCV cam -> POV-Ray location / look_at / sky / angle."""
    R = qvec_to_rotmat(qvec)
    C = camera_center(qvec, tvec)
    # Camera +Z in world (view direction), OpenCV Y down -> POV sky = -cam_Y
    forward = R[2, :]
    up = -R[1, :]
    look_at = C + forward
    return {
        "location": C,
        "look_at": look_at,
        "sky": up,
        "angle": fov_x_deg(fx, width),
        "right_ratio": width / float(height),
    }


def _fmt_vec(v: np.ndarray) -> str:
    return f"<{float(v[0]):.6f}, {float(v[1]):.6f}, {float(v[2]):.6f}>"


def write_cameras_inc(
    path: Path,
    frames: List[dict],
) -> None:
    lines = [
        "// Auto-generated COLMAP cameras for POV-Ray",
        f"#declare N_CAMERAS = {len(frames)};",
        "",
    ]
    for i, fr in enumerate(frames):
        cam = fr["pov"]
        lines += [
            f"#macro Cam_{i}()",
            "camera {",
            f"  location {_fmt_vec(cam['location'])}",
            f"  look_at {_fmt_vec(cam['look_at'])}",
            f"  sky {_fmt_vec(cam['sky'])}",
            f"  right x*{cam['right_ratio']:.6f}",
            f"  angle {cam['angle']:.4f}",
            "}",
            "#end",
            "",
        ]
    # Selector macro
    lines += [
        "#macro UseCamera(I)",
        "  #switch (I)",
    ]
    for i in range(len(frames)):
        lines.append(f"    #case ({i}) Cam_{i}() #break")
    lines += [
        "    #else Cam_0()",
        "  #end",
        "#end",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_points_inc(path: Path, points: dict, max_points: int = 20000, radius_scale: float = 0.15) -> int:
    pts = list(points.values())
    if len(pts) > max_points:
        sel = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = [pts[i] for i in sel]
    xyz = np.stack([p.xyz for p in pts])
    # radius from median knn-ish spacing
    if len(xyz) > 8:
        d = np.linalg.norm(xyz[1:51] - xyz[0], axis=1) if len(xyz) > 50 else np.linalg.norm(xyz[1:] - xyz[0], axis=1)
        r = float(np.median(d) * radius_scale)
    else:
        r = 0.05
    r = max(r, 1e-3)
    lines = [
        "// Sparse COLMAP points as spheres",
        "union {",
    ]
    for p in pts:
        c = p.rgb.astype(np.float64) / 255.0
        lines.append(
            f"  sphere {{ {_fmt_vec(p.xyz)}, {r:.5f} "
            f"pigment {{ rgb <{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}> }} "
            f"finish {{ ambient 0.35 diffuse 0.65 }} }}"
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(pts)


def write_mesh_inc(path: Path, mesh_path: Path, max_faces: int = 200000) -> Tuple[int, int]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"Empty mesh: {mesh_path}")
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    cols = np.asarray(mesh.vertex_colors)
    if faces.shape[0] > max_faces:
        # deterministic stride subsample of faces
        step = int(math.ceil(faces.shape[0] / max_faces))
        faces = faces[::step]
    n_v, n_f = verts.shape[0], faces.shape[0]

    lines = [
        f"// Mesh from {mesh_path.name}: {n_v} verts, {n_f} faces",
        "mesh2 {",
        f"  vertex_vectors {{ {n_v},",
    ]
    for v in verts:
        lines.append(f"    {_fmt_vec(v)},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("  }")
    lines.append(f"  texture_list {{ {n_v},")
    for c in cols:
        lines.append(
            f"    texture {{ pigment {{ rgb <{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}> }} "
            f"finish {{ ambient 0.25 diffuse 0.7 specular 0.05 }} }},"
        )
    lines[-1] = lines[-1].rstrip(",")
    lines.append("  }")
    lines.append(f"  face_indices {{ {n_f},")
    for f in faces:
        i, j, k = int(f[0]), int(f[1]), int(f[2])
        lines.append(f"    <{i}, {j}, {k}>, {i}, {j}, {k},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("  }")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_v, n_f


def write_photo_billboards_inc(
    path: Path,
    frames: List[dict],
    images_dir: Path,
    out_tex_dir: Path,
    every_n: int = 5,
    plane_scale: float = 1.0,
) -> int:
    """Optional: textured camera-facing planes (image projectors) as coarse proxy."""
    out_tex_dir.mkdir(parents=True, exist_ok=True)
    lines = ["// Textured billboards at selected camera frustums", "union {"]
    count = 0
    for i, fr in enumerate(frames):
        if i % every_n != 0:
            continue
        # Copy/resize texture for POV
        src = images_dir / fr["name"]
        tex_name = f"tex_{i:03d}.png"
        tex_path = out_tex_dir / tex_name
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((1024, 1024))
            im.save(tex_path)
        cam = fr["pov"]
        loc = cam["location"]
        look = cam["look_at"]
        fwd = look - loc
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        # Place plane slightly in front of camera along view
        dist = fr.get("billboard_dist", 5.0) * plane_scale
        center = loc + fwd * dist
        # Plane basis
        up = cam["sky"]
        up = up / (np.linalg.norm(up) + 1e-9)
        right = np.cross(fwd, up)
        nrm = np.linalg.norm(right)
        if nrm < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / nrm
        up = np.cross(right, fwd)
        up = up / (np.linalg.norm(up) + 1e-9)
        # POV plane is infinite in xy; use box or triangle pair - use polygon
        hw = dist * math.tan(math.radians(cam["angle"] * 0.5))
        hh = hw / cam["right_ratio"]
        c00 = center - right * hw - up * hh
        c10 = center + right * hw - up * hh
        c11 = center + right * hw + up * hh
        c01 = center - right * hw + up * hh
        rel = f"textures/{tex_name}"
        lines += [
            "  mesh2 {",
            "    vertex_vectors { 4,",
            f"      {_fmt_vec(c00)}, {_fmt_vec(c10)}, {_fmt_vec(c11)}, {_fmt_vec(c01)}",
            "    }",
            "    uv_vectors { 4, <0,0>, <1,0>, <1,1>, <0,1> }",
            "    face_indices { 2, <0,1,2>, <0,2,3> }",
            "    uv_indices { 2, <0,1,2>, <0,2,3> }",
            "    texture {",
            "      uv_mapping",
            f"      pigment {{ image_map {{ png \"{rel}\" interpolate 2 }} }}",
            "      finish { ambient 1.0 diffuse 0.0 }",
            "    }",
            "  }",
        ]
        count += 1
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def write_master_pov(
    path: Path,
    width: int,
    height: int,
    camera_index: int,
    use_mesh: bool,
    use_points: bool,
    use_billboards: bool,
) -> None:
    parts = [
        "// Blackberry / COLMAP scene for POV-Ray",
        "// Render example:",
        f"//   povray +W{width} +H{height} +A0.3 -D Declare=CAM_INDEX={camera_index} scene.pov",
        "",
        '#version 3.7;',
        "global_settings { assumed_gamma 1.0 }",
        "background { color rgb <0.02, 0.02, 0.03> }",
        "",
        '#include "cameras.inc"',
        "",
        "#ifndef (CAM_INDEX)",
        f"  #declare CAM_INDEX = {camera_index};",
        "#end",
        "UseCamera(CAM_INDEX)",
        "",
        "// Soft key + fill",
        "light_source { <50, 80, -40> color rgb 1.2 }",
        "light_source { <-30, 40, 60> color rgb 0.45 }",
        "",
        "// OpenCV/COLMAP -> optional axis flip for POV Y-up feel",
        "#declare WorldXform = transform { scale <1, -1, -1> };",
        "",
        "object {",
    ]
    if use_mesh:
        parts += ['  #include "geometry_mesh.inc"']
    if use_points and not use_mesh:
        parts += ['  #include "geometry_points.inc"']
    if use_billboards:
        parts += ['  #include "billboards.inc"']
    if not use_mesh and not use_points and not use_billboards:
        parts += [
            "  // Fallback marker",
            "  sphere { <0,0,0>, 0.2 pigment { rgb <1,0,0> } }",
        ]
    parts += [
        "  transform { WorldXform }",
        "}",
        "",
    ]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def load_frames(data_root: Path) -> Tuple[List[dict], dict]:
    images_dir = data_root / "images"
    cameras = read_cameras_text(data_root / "cameras.txt")
    images = read_images_text(data_root / "images.txt", load_points2d=False)
    ordered = sorted(images.values(), key=lambda im: im.image_id)
    w, h = _probe_wh(images_dir, ordered[0].name)
    cameras = {cid: scale_camera_to_image_size(cam, w, h) for cid, cam in cameras.items()}
    frames = []
    for im in ordered:
        cam = cameras[im.camera_id]
        K, _ = camera_K_dist(cam)
        pov = colmap_camera_to_pov(im.qvec, im.tvec, float(K[0, 0]), cam.width, cam.height)
        frames.append(
            {
                "image_id": im.image_id,
                "name": im.name,
                "width": cam.width,
                "height": cam.height,
                "pov": pov,
                "billboard_dist": float(np.linalg.norm(pov["location"])) * 0.02 + 2.0,
            }
        )
    return frames, {"width": w, "height": h}


def main() -> None:
    ap = argparse.ArgumentParser(description="COLMAP + mesh/points -> POV-Ray scene")
    ap.add_argument(
        "--data",
        type=Path,
        default=Path.home() / "tmp" / "blackberry-splats" / "dataset",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO / "povray_out",
    )
    ap.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Optional triangle mesh (ply/obj) with vertex colors",
    )
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--max-points", type=int, default=25000)
    ap.add_argument("--max-faces", type=int, default=150000)
    ap.add_argument("--billboards", action="store_true", help="Add textured camera planes")
    ap.add_argument("--billboard-every", type=int, default=8)
    ap.add_argument("--no-points", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, meta = load_frames(Path(args.data))
    write_cameras_inc(out / "cameras.inc", frames)

    use_mesh = False
    use_points = False
    mesh_path = args.mesh
    if mesh_path is None:
        # Prefer workspace engines_out mesh if present
        cand = _REPO / "engines_out" / "mesh.ply"
        if cand.is_file():
            mesh_path = cand
    if mesh_path is not None and Path(mesh_path).is_file():
        try:
            nv, nf = write_mesh_inc(out / "geometry_mesh.inc", Path(mesh_path), max_faces=args.max_faces)
            print(f"Wrote mesh: {nv} verts, {nf} faces")
            use_mesh = True
        except Exception as e:
            print(f"Mesh export failed ({e}), falling back to points")

    if not use_mesh and not args.no_points:
        points = read_points3d_text(Path(args.data) / "points3D.txt")
        n = write_points_inc(out / "geometry_points.inc", points, max_points=args.max_points)
        print(f"Wrote points: {n} spheres")
        use_points = True

    use_bb = False
    if args.billboards:
        nbb = write_photo_billboards_inc(
            out / "billboards.inc",
            frames,
            Path(args.data) / "images",
            out / "textures",
            every_n=args.billboard_every,
        )
        print(f"Wrote billboards: {nbb}")
        use_bb = True

    write_master_pov(
        out / "scene.pov",
        meta["width"],
        meta["height"],
        args.camera_index,
        use_mesh=use_mesh,
        use_points=use_points,
        use_billboards=use_bb,
    )
    # Camera name map
    (out / "cameras.json").write_text(
        __import__("json").dumps(
            [{"index": i, "name": fr["name"], "image_id": fr["image_id"]} for i, fr in enumerate(frames)],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Scene ready in {out}")
    print(f"  cameras: {len(frames)}")
    print(f"  render: povray +W{meta['width']} +H{meta['height']} +A0.3 Declare=CAM_INDEX=0 {out/'scene.pov'}")


if __name__ == "__main__":
    main()
