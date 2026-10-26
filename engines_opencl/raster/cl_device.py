"""OpenCL platform/device discovery (Intel Arc, AMD, NVIDIA via OpenCL ICD)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CLDeviceInfo:
    platform: str
    device: str
    device_type: str
    index: int


def list_devices() -> List[CLDeviceInfo]:
    try:
        import pyopencl as cl
    except ImportError as e:
        raise ImportError("pyopencl is required: pip install pyopencl") from e

    out: List[CLDeviceInfo] = []
    try:
        platforms = cl.get_platforms()
    except Exception:
        return out
    idx = 0
    for p in platforms:
        for d in p.get_devices():
            t = cl.device_type.to_string(d.type)
            out.append(CLDeviceInfo(p.name.strip(), d.name.strip(), t, idx))
            idx += 1
    return out


def make_context(prefer_gpu: bool = True, device_index: Optional[int] = None):
    import pyopencl as cl

    devices = list_devices()
    if not devices:
        raise RuntimeError(
            "No OpenCL platforms found. Install a GPU ICD "
            "(Intel Compute Runtime / NEO for Arc, ROCm OpenCL, or NVIDIA OpenCL)."
        )
    if device_index is not None:
        # Rebuild selection by flat index
        flat = []
        for p in cl.get_platforms():
            for d in p.get_devices():
                flat.append((p, d))
        if device_index < 0 or device_index >= len(flat):
            raise RuntimeError(f"device_index {device_index} out of range 0..{len(flat)-1}")
        p, d = flat[device_index]
        ctx = cl.Context(devices=[d])
        return ctx, cl.CommandQueue(ctx), d

    # Prefer GPU then ACCELERATOR then CPU
    order = []
    for p in cl.get_platforms():
        for d in p.get_devices():
            order.append((p, d))
    def score(pair):
        _, d = pair
        if d.type & cl.device_type.GPU:
            return 0
        if d.type & cl.device_type.ACCELERATOR:
            return 1
        return 2
    order.sort(key=score)
    p, d = order[0]
    ctx = cl.Context(devices=[d])
    return ctx, cl.CommandQueue(ctx), d
