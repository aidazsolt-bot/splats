"""Load blackberry-style COLMAP+RGBA dataset and scale intrinsics to image size."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .colmap_io import (
    Camera,
    ImagePose,
    Point3D,
    camera_K_dist,
    camera_center,
    qvec_to_rotmat,
    read_cameras_text,
    read_images_text,
    read_points3d_text,
    scale_camera_to_image_size,
    write_cameras_text,
    write_images_text,
    write_points3d_text,
)


@dataclass
class ViewData:
    image_id: int
    name: str
    path: Path
    qvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray  # world-to-camera
    C: np.ndarray  # camera center in world
    camera_id: int
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int


@dataclass
class Dataset:
    root: Path
    images_dir: Path
    cameras: Dict[int, Camera]
    images: Dict[int, ImagePose]
    points: Dict[int, Point3D]
    views: List[ViewData]
    image_wh: Tuple[int, int]


def _probe_image_size(images_dir: Path, name: str) -> Tuple[int, int]:
    path = images_dir / name
    with Image.open(path) as im:
        w, h = im.size
    return w, h


def load_dataset(
    root: Path,
    images_subdir: str = "images",
    scale_intrinsics: bool = True,
    max_views: Optional[int] = None,
) -> Dataset:
    root = Path(root)
    images_dir = root / images_subdir
    cameras = read_cameras_text(root / "cameras.txt")
    images = read_images_text(root / "images.txt", load_points2d=False)
    points = read_points3d_text(root / "points3D.txt")

    # Determine image resolution from first available file
    ordered = sorted(images.values(), key=lambda im: im.image_id)
    if max_views is not None:
        ordered = ordered[:max_views]
    if not ordered:
        raise RuntimeError("No images in COLMAP model")
    img_w, img_h = _probe_image_size(images_dir, ordered[0].name)

    if scale_intrinsics:
        cameras = {
            cid: scale_camera_to_image_size(cam, img_w, img_h) for cid, cam in cameras.items()
        }

    views: List[ViewData] = []
    for im in ordered:
        cam = cameras[im.camera_id]
        K, dist = camera_K_dist(cam)
        R = qvec_to_rotmat(im.qvec)
        C = camera_center(im.qvec, im.tvec)
        views.append(
            ViewData(
                image_id=im.image_id,
                name=im.name,
                path=images_dir / im.name,
                qvec=im.qvec,
                tvec=im.tvec,
                R=R,
                C=C,
                camera_id=im.camera_id,
                K=K,
                dist=dist,
                width=cam.width,
                height=cam.height,
            )
        )

    return Dataset(
        root=root,
        images_dir=images_dir,
        cameras=cameras,
        images={v.image_id: images[v.image_id] for v in views if v.image_id in images},
        points=points,
        views=views,
        image_wh=(img_w, img_h),
    )


def save_scaled_colmap(dataset: Dataset, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_cameras_text(out_dir / "cameras.txt", dataset.cameras)
    write_images_text(out_dir / "images.txt", dataset.images)
    write_points3d_text(out_dir / "points3D.txt", dataset.points)


def load_rgba(path: Path, downscale: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Load RGB uint8 HxWx3 and alpha float32 HxW in [0,1]."""
    im = Image.open(path).convert("RGBA")
    arr = np.asarray(im)
    if downscale > 1:
        new_w = max(1, arr.shape[1] // downscale)
        new_h = max(1, arr.shape[0] // downscale)
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = arr[:, :, :3].astype(np.uint8)
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    return rgb, alpha


def scaled_K(K: np.ndarray, downscale: int) -> np.ndarray:
    if downscale <= 1:
        return K.copy()
    s = 1.0 / float(downscale)
    K2 = K.copy()
    K2[0, 0] *= s
    K2[1, 1] *= s
    K2[0, 2] *= s
    K2[1, 2] *= s
    return K2
