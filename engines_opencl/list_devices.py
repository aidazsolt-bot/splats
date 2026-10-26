#!/usr/bin/env python3
"""Print OpenCL platforms/devices."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from raster.cl_device import list_devices  # noqa: E402


def main() -> None:
    try:
        devs = list_devices()
    except ImportError as e:
        print(e)
        sys.exit(1)
    if not devs:
        print("No OpenCL devices found.")
        print("Intel Arc: install Intel Compute Runtime / oneAPI Level Zero + OpenCL ICD.")
        print("AMD: ROCm OpenCL. NVIDIA: NVIDIA OpenCL ICD (alongside CUDA).")
        sys.exit(2)
    for d in devs:
        print(f"[{d.index}] {d.platform} | {d.device} | {d.device_type}")


if __name__ == "__main__":
    main()
