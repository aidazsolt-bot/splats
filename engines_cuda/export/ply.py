"""Export trained Gaussians to 3DGS / Postshot-compatible PLY."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@torch.no_grad()
def export_gsplat_ply(
    path: Path,
    means: torch.Tensor,
    scales_log: torch.Tensor,
    quats: torch.Tensor,
    opacities_logit: torch.Tensor,
    colors_dc: torch.Tensor,
    colors_rest: Optional[torch.Tensor] = None,
) -> Path:
    """Write binary_little_endian PLY with SH degree 3 layout (45 rest coeffs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    means_np = means.detach().float().cpu().numpy()
    scales_np = scales_log.detach().float().cpu().numpy()
    quats_np = quats.detach().float().cpu().numpy()
    # normalize quats
    quats_np = quats_np / (np.linalg.norm(quats_np, axis=-1, keepdims=True) + 1e-12)
    opac_np = opacities_logit.detach().float().cpu().numpy().reshape(-1)
    dc_np = colors_dc.detach().float().cpu().numpy()
    n = means_np.shape[0]

    rest = np.zeros((n, 45), dtype=np.float32)
    if colors_rest is not None:
        cr = colors_rest.detach().float().cpu().numpy()
        # (N, K, 3) -> flatten as f_rest_i for i in 0..44 matching 3DGS (band-major RGB)
        # 3DGS stores rest as (N, 3, (d+1)^2-1) then transpose; common export flattens K*3
        k = min(cr.shape[1], 15)
        # flatten as rest_{i*3+c}
        flat = cr[:, :k, :].reshape(n, -1)
        rest[:, : flat.shape[1]] = flat

    normals = np.zeros((n, 3), dtype=np.float32)

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(45):
        header.append(f"property float f_rest_{i}")
    header += [
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]

    with path.open("wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        for i in range(n):
            vals = [
                float(means_np[i, 0]),
                float(means_np[i, 1]),
                float(means_np[i, 2]),
                float(normals[i, 0]),
                float(normals[i, 1]),
                float(normals[i, 2]),
                float(dc_np[i, 0]),
                float(dc_np[i, 1]),
                float(dc_np[i, 2]),
            ]
            vals.extend(float(x) for x in rest[i])
            vals.append(float(opac_np[i]))
            vals.extend(float(x) for x in scales_np[i])
            vals.extend(float(x) for x in quats_np[i])
            f.write(struct.pack("<62f", *vals))
    return path
