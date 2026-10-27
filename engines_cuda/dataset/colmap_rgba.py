"""COLMAP text + RGBA torch dataset with intrinsic scaling and alpha masks."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_REPO_ENGINES = Path(__file__).resolve().parents[2] / "engines"
if str(_REPO_ENGINES) not in sys.path:
    sys.path.insert(0, str(_REPO_ENGINES))

from prep.colmap_io import (  # noqa: E402
    camera_K_dist,
    qvec_to_rotmat,
    read_cameras_text,
    read_images_text,
    read_points3d_text,
    scale_camera_to_image_size,
)


def _rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0,1] -> SH DC (N,3)."""
    c0 = 0.28209479177387814
    return ((rgb - 0.5) / c0).astype(np.float32)


@dataclass
class SceneInit:
    means: torch.Tensor  # (N,3)
    scales: torch.Tensor  # (N,3) linear
    quats: torch.Tensor  # (N,4) wxyz
    opacities: torch.Tensor  # (N,) logit
    colors_dc: torch.Tensor  # (N,3) SH DC
    colors_rest: Optional[torch.Tensor]  # (N, K, 3) or None
    scene_scale: float


class ColmapRGBADataset(Dataset):
    """One item = dict with image, alpha, K, viewmat (c2w world-to-cam 4x4), size."""

    def __init__(
        self,
        root: Path,
        images_subdir: str = "images",
        data_factor: int = 1,
        split: str = "train",
        test_every: int = 8,
        resize_to: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.root = Path(root)
        self.images_dir = self.root / images_subdir
        self.data_factor = max(1, int(data_factor))
        self.split = split

        cameras = read_cameras_text(self.root / "cameras.txt")
        images = read_images_text(self.root / "images.txt", load_points2d=False)
        self.points = read_points3d_text(self.root / "points3D.txt")

        ordered = sorted(images.values(), key=lambda im: im.image_id)
        # Probe resolution
        with Image.open(self.images_dir / ordered[0].name) as im0:
            img_w, img_h = im0.size
        cameras = {
            cid: scale_camera_to_image_size(cam, img_w, img_h) for cid, cam in cameras.items()
        }

        self.frames: List[dict] = []
        for i, im in enumerate(ordered):
            is_test = test_every > 0 and (i % test_every == 0)
            if split == "train" and is_test:
                continue
            if split == "test" and not is_test:
                continue
            cam = cameras[im.camera_id]
            K, dist = camera_K_dist(cam)
            R = qvec_to_rotmat(im.qvec)  # world -> cam
            t = im.tvec.astype(np.float64)
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = R
            w2c[:3, 3] = t
            self.frames.append(
                {
                    "path": self.images_dir / im.name,
                    "name": im.name,
                    "K": K.astype(np.float32),
                    "dist": dist.astype(np.float32),
                    "w2c": w2c.astype(np.float32),
                    "width": cam.width,
                    "height": cam.height,
                }
            )

        self.resize_to = resize_to
        self.cameras = cameras
        self._cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.frames)

    def _load_rgba_u8(self, path: Path) -> np.ndarray:
        with Image.open(path) as src:
            im = src.convert("RGBA")
        if self.data_factor > 1:
            w, h = im.size
            im = im.resize(
                (max(1, w // self.data_factor), max(1, h // self.data_factor)),
                Image.BICUBIC,
            )
        if self.resize_to is not None:
            im = im.resize(self.resize_to, Image.BICUBIC)
        return np.ascontiguousarray(np.asarray(im, dtype=np.uint8))

    def preload(self, workers: int = 8, label: str = "") -> None:
        """Decode + downscale all images once into RAM (uint8) so training never re-reads disk/network."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        todo = [i for i in range(len(self.frames)) if i not in self._cache]
        if not todo:
            return
        tag = f"[{label}] " if label else ""
        total = len(todo)
        nbytes = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(self._load_rgba_u8, self.frames[i]["path"]): i for i in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                arr = fut.result()
                self._cache[futs[fut]] = arr
                nbytes += arr.nbytes
                if n == total or n % 10 == 0:
                    print(f"  {tag}cached {n}/{total} images ({nbytes / 2**20:.0f} MiB)", flush=True)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fr = self.frames[idx]
        u8 = self._cache.get(idx)
        if u8 is None:
            u8 = self._load_rgba_u8(fr["path"])
        arr = u8.astype(np.float32) / 255.0
        rgb, alpha = arr[..., :3], arr[..., 3]
        h, w = rgb.shape[:2]
        K = fr["K"].copy()
        if self.data_factor > 1 or self.resize_to is not None:
            sx = w / float(fr["width"])
            sy = h / float(fr["height"])
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy
        return {
            "image": torch.from_numpy(rgb),  # H W 3
            "alpha": torch.from_numpy(alpha),  # H W
            "K": torch.from_numpy(K),
            "viewmat": torch.from_numpy(fr["w2c"]),  # world to camera
            "height": torch.tensor(h, dtype=torch.int32),
            "width": torch.tensor(w, dtype=torch.int32),
            "name": fr["name"],
        }


def init_gaussians_from_points3d(
    points: dict,
    device: torch.device,
    sh_degree: int = 3,
    init_opacity: float = 0.1,
    scale_factor: float = 1.0,
) -> SceneInit:
    """SfM init: means + colors from COLMAP; isotropic scales from knn."""
    pts = list(points.values())
    xyz = np.stack([p.xyz for p in pts], axis=0).astype(np.float32)
    rgb = np.stack([p.rgb for p in pts], axis=0).astype(np.float32) / 255.0

    # knn scale
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(xyz)
        dists, _ = tree.query(xyz, k=min(4, len(xyz)))
        if dists.ndim == 1:
            nn = np.maximum(dists, 1e-6)
        else:
            nn = np.maximum(dists[:, 1], 1e-6)
        scales = np.repeat((nn * scale_factor)[:, None], 3, axis=1).astype(np.float32)
    except Exception:
        scales = np.full((len(xyz), 3), 0.01 * scale_factor, dtype=np.float32)

    quats = np.zeros((len(xyz), 4), dtype=np.float32)
    quats[:, 0] = 1.0
    opacities = np.full((len(xyz),), init_opacity, dtype=np.float32)
    # logit
    opacities = np.log(opacities / (1.0 - opacities))
    colors_dc = _rgb_to_sh_dc(rgb)
    n_rest = (sh_degree + 1) ** 2 - 1
    colors_rest = np.zeros((len(xyz), n_rest, 3), dtype=np.float32) if n_rest > 0 else None

    extent = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    scene_scale = max(extent, 1e-3)

    return SceneInit(
        means=torch.from_numpy(xyz).to(device),
        scales=torch.from_numpy(np.log(np.maximum(scales, 1e-8))).to(device),
        quats=torch.from_numpy(quats).to(device),
        opacities=torch.from_numpy(opacities).to(device),
        colors_dc=torch.from_numpy(colors_dc).to(device),
        colors_rest=torch.from_numpy(colors_rest).to(device) if colors_rest is not None else None,
        scene_scale=scene_scale,
    )
