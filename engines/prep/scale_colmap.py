#!/usr/bin/env python3
"""Scale COLMAP cameras.txt intrinsics to match PNG resolution; write prep outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ENGINES = Path(__file__).resolve().parents[1]
if str(_ENGINES) not in sys.path:
    sys.path.insert(0, str(_ENGINES))

from prep.dataset import load_dataset, save_scaled_colmap  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Scale COLMAP intrinsics to image size")
    ap.add_argument("--data", type=Path, required=True, help="Dataset root with cameras.txt and images/")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for scaled COLMAP texts")
    args = ap.parse_args()

    ds = load_dataset(args.data, scale_intrinsics=True)
    save_scaled_colmap(ds, args.out)
    cam = next(iter(ds.cameras.values()))
    print(
        f"Scaled cameras to {cam.width}x{cam.height}, model={cam.model}, "
        f"n_images={len(ds.views)}, n_points={len(ds.points)}"
    )


if __name__ == "__main__":
    main()
