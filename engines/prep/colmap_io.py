"""COLMAP text model I/O with intrinsic scaling to match image resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # model-dependent


@dataclass
class ImagePose:
    image_id: int
    qvec: np.ndarray  # qw, qx, qy, qz
    tvec: np.ndarray  # tx, ty, tz
    camera_id: int
    name: str
    # Optional 2D observations: (N, 3) as x, y, point3d_id (-1 if none)
    xys: Optional[np.ndarray] = None
    point3d_ids: Optional[np.ndarray] = None


@dataclass
class Point3D:
    point_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    # track: list of (image_id, point2d_idx)
    track: List[Tuple[int, int]]


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (qw,qx,qy,qz) to rotation matrix (world-from-camera R such that x_cam = R x_world + t)."""
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def camera_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R = qvec_to_rotmat(qvec)
    return (-R.T @ tvec).astype(np.float64)


def read_cameras_text(path: Path) -> Dict[int, Camera]:
    cams: Dict[int, Camera] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            cam_id = int(toks[0])
            model = toks[1]
            width = int(toks[2])
            height = int(toks[3])
            params = np.array([float(x) for x in toks[4:]], dtype=np.float64)
            cams[cam_id] = Camera(cam_id, model, width, height, params)
    return cams


def read_images_text(path: Path, load_points2d: bool = False) -> Dict[int, ImagePose]:
    """COLMAP images.txt: two lines per image (pose then POINTS2D)."""
    images: Dict[int, ImagePose] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        toks = line.split()
        if len(toks) < 10:
            i += 1
            continue
        image_id = int(toks[0])
        qvec = np.array([float(x) for x in toks[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in toks[5:8]], dtype=np.float64)
        camera_id = int(toks[8])
        name = toks[9]
        xys = None
        point3d_ids = None
        # Always consume the following POINTS2D line when present.
        if i + 1 < len(lines):
            obs_line = lines[i + 1].strip()
            if obs_line and not obs_line.startswith("#"):
                obs = obs_line.split()
                # Pose lines are exactly IMAGE_ID ... NAME; POINTS2D have 3*N tokens.
                looks_like_pose = len(obs) >= 10 and obs[-1].lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
                )
                if not looks_like_pose:
                    if load_points2d and obs:
                        vals = np.array([float(x) for x in obs], dtype=np.float64)
                        if vals.size >= 3 and vals.size % 3 == 0:
                            pts = vals.reshape(-1, 3)
                            xys = pts[:, :2]
                            point3d_ids = pts[:, 2].astype(np.int64)
                    i += 1
        images[image_id] = ImagePose(image_id, qvec, tvec, camera_id, name, xys, point3d_ids)
        i += 1
    return images


def read_points3d_text(path: Path) -> Dict[int, Point3D]:
    points: Dict[int, Point3D] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            pid = int(toks[0])
            xyz = np.array([float(toks[1]), float(toks[2]), float(toks[3])], dtype=np.float64)
            rgb = np.array([int(toks[4]), int(toks[5]), int(toks[6])], dtype=np.uint8)
            error = float(toks[7])
            track: List[Tuple[int, int]] = []
            rest = toks[8:]
            for j in range(0, len(rest), 2):
                if j + 1 >= len(rest):
                    break
                track.append((int(rest[j]), int(rest[j + 1])))
            points[pid] = Point3D(pid, xyz, rgb, error, track)
    return points


def scale_camera_to_image_size(cam: Camera, img_w: int, img_h: int) -> Camera:
    """Scale SIMPLE_PINHOLE / SIMPLE_RADIAL / PINHOLE / OPENCV params to target resolution."""
    sx = img_w / float(cam.width)
    sy = img_h / float(cam.height)
    if abs(sx - sy) > 1e-3:
        # Non-uniform: use geometric mean scale for focal, separate for principal point
        pass
    params = cam.params.copy()
    model = cam.model.upper()
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        # f, cx, cy [, k]
        params[0] *= sx  # approximate with sx (images share aspect)
        params[1] *= sx
        params[2] *= sy
        # distortion k unchanged for SIMPLE_RADIAL (dimensionless in COLMAP normalized coords)
    elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
        # fx, fy, cx, cy, ...
        params[0] *= sx
        params[1] *= sy
        params[2] *= sx
        params[3] *= sy
    else:
        # Best-effort: scale first four like PINHOLE if long enough
        if params.size >= 4:
            params[0] *= sx
            params[1] *= sy
            params[2] *= sx
            params[3] *= sy
        elif params.size >= 3:
            params[0] *= sx
            params[1] *= sx
            params[2] *= sy
    return Camera(cam.camera_id, cam.model, img_w, img_h, params)


def camera_K_dist(cam: Camera) -> Tuple[np.ndarray, np.ndarray]:
    """Return 3x3 K and distortion coeffs for OpenCV (k1,k2,p1,p2[,k3])."""
    model = cam.model.upper()
    p = cam.params
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k, 0, 0, 0, 0], dtype=np.float64)
    elif model == "PINHOLE":
        fx, fy, cx, cy = p
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, 0], dtype=np.float64)
    else:
        # Fallback SIMPLE_RADIAL-like
        f, cx, cy = p[0], p[1], p[2]
        k = p[3] if p.size > 3 else 0.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k, 0, 0, 0, 0], dtype=np.float64)
    return K, dist


def write_cameras_text(path: Path, cameras: Dict[int, Camera]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    for cam_id in sorted(cameras):
        c = cameras[cam_id]
        params = " ".join(repr(float(x)) for x in c.params)
        lines.append(f"{c.camera_id} {c.model} {c.width} {c.height} {params}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_images_text(path: Path, images: Dict[int, ImagePose]) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(images)}",
    ]
    for image_id in sorted(images):
        im = images[image_id]
        q = " ".join(repr(float(x)) for x in im.qvec)
        t = " ".join(repr(float(x)) for x in im.tvec)
        lines.append(f"{im.image_id} {q} {t} {im.camera_id} {im.name}")
        lines.append("")  # empty POINTS2D (poses only)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_points3d_text(path: Path, points: Dict[int, Point3D]) -> None:
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {len(points)}",
    ]
    for pid in sorted(points):
        p = points[pid]
        track = " ".join(f"{a} {b}" for a, b in p.track)
        lines.append(
            f"{p.point_id} {p.xyz[0]} {p.xyz[1]} {p.xyz[2]} "
            f"{int(p.rgb[0])} {int(p.rgb[1])} {int(p.rgb[2])} {p.error} {track}".rstrip()
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
