"""OpenCL platform/device discovery (Intel Arc, AMD, NVIDIA via OpenCL ICD)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CLDeviceInfo:
    platform: str
    device: str
    device_type: str
    index: int


def _flat_devices():
    import pyopencl as cl

    flat = []
    for p in cl.get_platforms():
        for d in p.get_devices():
            flat.append((p, d))
    return flat


def list_devices() -> List[CLDeviceInfo]:
    try:
        import pyopencl as cl
    except ImportError as e:
        raise ImportError("pyopencl is required: pip install pyopencl") from e

    out: List[CLDeviceInfo] = []
    try:
        flat = _flat_devices()
    except Exception:
        return out
    for idx, (p, d) in enumerate(flat):
        t = cl.device_type.to_string(d.type)
        out.append(CLDeviceInfo(p.name.strip(), d.name.strip(), t, idx))
    return out


def resolve_device_index(
    device_index: Optional[int] = None,
    prefer: Optional[str] = None,
) -> int:
    """Resolve flat device index. prefer: substring match e.g. 'nvidia', 'intel'."""
    flat = _flat_devices()
    if not flat:
        raise RuntimeError(
            "No OpenCL platforms found. Install a GPU ICD "
            "(Intel Compute Runtime / NEO for Arc, ROCm OpenCL, or NVIDIA OpenCL)."
        )
    if prefer:
        key = prefer.lower()
        for i, (p, d) in enumerate(flat):
            blob = f"{p.name} {d.name}".lower()
            if key in blob:
                return i
        names = [f"[{i}] {p.name} | {d.name}" for i, (p, d) in enumerate(flat)]
        raise RuntimeError(f"No OpenCL device matching prefer={prefer!r}. Have:\n" + "\n".join(names))
    if device_index is None:
        # Prefer NVIDIA, then non-Intel GPU, then any GPU
        import pyopencl as cl

        def score(i_pd):
            i, (p, d) = i_pd
            name = f"{p.name} {d.name}".lower()
            is_gpu = bool(d.type & cl.device_type.GPU)
            if "nvidia" in name or "geforce" in name or "quadro" in name:
                return (0, i)
            if is_gpu and "intel" not in name and "uhd" not in name:
                return (1, i)
            if is_gpu:
                return (2, i)
            return (3, i)

        return min(range(len(flat)), key=lambda i: score((i, flat[i])))
    if device_index < 0 or device_index >= len(flat):
        raise RuntimeError(f"device_index {device_index} out of range 0..{len(flat)-1}")
    return device_index


def make_context(
    prefer_gpu: bool = True,
    device_index: Optional[int] = None,
    prefer: Optional[str] = None,
) -> Tuple:
    import pyopencl as cl

    idx = resolve_device_index(device_index=device_index, prefer=prefer)
    flat = _flat_devices()
    p, d = flat[idx]
    # Bind explicitly to the platform - required on multi-ICD Windows setups
    props = [(cl.context_properties.PLATFORM, p)]
    ctx = cl.Context(devices=[d], properties=props)
    queue = cl.CommandQueue(ctx)
    print(f"OpenCL selected [{idx}]: platform={p.name.strip()} device={d.name.strip()}")
    return ctx, queue, d, idx
