"""PyOpenCL Gaussian rasterizer (forward)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .cl_device import make_context

_CL_SRC = (Path(__file__).resolve().parents[1] / "cl" / "rasterize.cl").read_text(encoding="utf-8")


class OpenCLGaussianRasterizer:
    def __init__(self, device_index: Optional[int] = None) -> None:
        import pyopencl as cl

        self.cl = cl
        self.ctx, self.queue, self.device = make_context(device_index=device_index)
        self.prg = cl.Program(self.ctx, _CL_SRC).build()
        print(f"OpenCL device: {self.device.name} ({cl.device_type.to_string(self.device.type)})")

    def rasterize(
        self,
        means: np.ndarray,  # N,3 float32
        scales_log: np.ndarray,
        quats: np.ndarray,
        opacities: np.ndarray,  # N sigmoid
        rgbs: np.ndarray,  # N,3
        viewmat: np.ndarray,  # 4,4
        K: np.ndarray,  # 3,3
        width: int,
        height: int,
        near: float = 0.01,
        max_radius: int = 28,
        depth_gamma: float = 4.0,
        fp_scale: float = 10000.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cl = self.cl
        mf = cl.mem_flags
        N = int(means.shape[0])
        npix = width * height

        means = np.ascontiguousarray(means, dtype=np.float32)
        scales_log = np.ascontiguousarray(scales_log, dtype=np.float32)
        quats = np.ascontiguousarray(quats, dtype=np.float32)
        opacities = np.ascontiguousarray(opacities, dtype=np.float32)
        rgbs = np.ascontiguousarray(rgbs, dtype=np.float32)
        viewmat = np.ascontiguousarray(viewmat.reshape(-1), dtype=np.float32)
        K = np.ascontiguousarray(K.reshape(-1), dtype=np.float32)

        d_means = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=means)
        d_scales = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=scales_log)
        d_quats = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=quats)
        d_view = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=viewmat)
        d_K = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K)
        d_m2 = cl.Buffer(self.ctx, mf.READ_WRITE, size=N * 2 * 4)
        d_c2 = cl.Buffer(self.ctx, mf.READ_WRITE, size=N * 4 * 4)
        d_dep = cl.Buffer(self.ctx, mf.READ_WRITE, size=N * 4)
        d_rad = cl.Buffer(self.ctx, mf.READ_WRITE, size=N * 4)
        d_val = cl.Buffer(self.ctx, mf.READ_WRITE, size=N)

        self.prg.project_gaussians(
            self.queue,
            (N,),
            None,
            d_means,
            d_scales,
            d_quats,
            d_view,
            d_K,
            d_m2,
            d_c2,
            d_dep,
            d_rad,
            d_val,
            np.int32(N),
            np.int32(width),
            np.int32(height),
            np.float32(near),
        )

        depths = np.empty(N, dtype=np.float32)
        cl.enqueue_copy(self.queue, depths, d_dep)
        valid = np.empty(N, dtype=np.uint8)
        cl.enqueue_copy(self.queue, valid, d_val)
        if not np.any(valid):
            return np.zeros((height, width, 3), np.float32), np.zeros((height, width), np.float32)
        d_min = float(depths[valid.astype(bool)].min())
        d_max = float(depths[valid.astype(bool)].max())
        d_range = max(d_max - d_min, 1e-3)

        d_op = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=opacities)
        d_rgb = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rgbs)
        d_acc_rgb = cl.Buffer(self.ctx, mf.READ_WRITE, size=npix * 3 * 4)
        d_acc_a = cl.Buffer(self.ctx, mf.READ_WRITE, size=npix * 4)

        self.prg.clear_accum(self.queue, (npix,), None, d_acc_rgb, d_acc_a, np.int32(npix))
        self.prg.splat_gaussians(
            self.queue,
            (N,),
            None,
            d_m2,
            d_c2,
            d_dep,
            d_rad,
            d_val,
            d_op,
            d_rgb,
            d_acc_rgb,
            d_acc_a,
            np.int32(N),
            np.int32(width),
            np.int32(height),
            np.float32(depth_gamma),
            np.float32(d_min),
            np.float32(d_range),
            np.int32(max_radius),
            np.float32(fp_scale),
        )

        d_out_rgb = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=npix * 3 * 4)
        d_out_a = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=npix * 4)
        self.prg.finalize_image(
            self.queue,
            (npix,),
            None,
            d_acc_rgb,
            d_acc_a,
            d_out_rgb,
            d_out_a,
            np.int32(npix),
            np.float32(fp_scale),
        )
        out_rgb = np.empty((npix, 3), dtype=np.float32)
        out_a = np.empty(npix, dtype=np.float32)
        cl.enqueue_copy(self.queue, out_rgb, d_out_rgb)
        cl.enqueue_copy(self.queue, out_a, d_out_a)
        self.queue.finish()
        return out_rgb.reshape(height, width, 3), out_a.reshape(height, width)
