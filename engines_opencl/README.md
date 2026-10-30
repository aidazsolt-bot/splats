# engines_opencl - 3DGS via OpenCL (Intel Arc / AMD / NVIDIA OpenCL)

OpenCL path for machines **without usable NVIDIA CUDA / gsplat** (e.g. Huawei
MateBook with **Intel Arc**). Training is a **hybrid** setup:

- **Train step (params need grad):** pure PyTorch CPU rasterizer
  (`engines_cpu_train` math - required for autograd)
- **No-grad preview:** OpenCL kernels (`cl/rasterize.cl`) when an ICD device
  is available

So OpenCL speeds **preview / inference-style** forwards. The Adam loop itself
still runs the CPU differentiable raster. Without an OpenCL ICD, pass
`--cpu-fallback` to train anyway.

## Setup

```bash
pip install -r engines_opencl/requirements.txt

# Always list devices first - note name substrings for --prefer
python engines_opencl/list_devices.py
```

`--prefer` matches a **case-insensitive substring** of the OpenCL device name
(default in CLI: `nvidia`). Use `--prefer ""` for first usable device, or
`--device-index N` for an exact index from `list_devices.py`.

`--data` Windows: `X:\blackberry-splats\dataset`  
`--data` Linux: `/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset`

## Command examples by GPU

### 1) Huawei MateBook / Intel Arc (primary target)

Install **Intel Graphics** drivers (Windows) or **Intel Compute Runtime / oneAPI
OpenCL ICD** (Linux). Confirm Arc appears in `list_devices.py` (not only UHD).

```bash
# short check (OpenCL required unless --cpu-fallback)
python engines_opencl/run_train.py --prefer arc \
  --data X:\blackberry-splats\dataset --out opencl_out_mate --smoke

# Colab-smoke-sized run (500 steps, factor 4) on MateBook
python engines_opencl/run_train.py --prefer arc \
  --data X:\blackberry-splats\dataset --out opencl_out_mate \
  --data-factor 4 --steps 500 --ckpt-every 50

# if list_devices shows "Intel(R) Arc(TM) ..." but not the word Arc, try:
python engines_opencl/run_train.py --prefer intel \
  --data X:\blackberry-splats\dataset --out opencl_out_mate \
  --data-factor 4 --steps 500
```

Dual GPU (Arc + UHD): prefer `arc` / a unique Arc substring so training does
not bind the weak UHD by accident. Or pin with `--device-index` from the list.

### 2) AMD (ROCm / RADV OpenCL ICD)

```bash
python engines_opencl/run_train.py --prefer amd \
  --data /path/to/dataset --out opencl_out_amd \
  --data-factor 4 --steps 5000
# alternatives if needed: --prefer radeon   or   --prefer gfx
```

### 3) NVIDIA - OpenCL only (old driver / no CUDA torch)

Prefer **`engines_cuda` + gsplat** when CUDA works. If you only have OpenCL:

```bash
python engines_opencl/run_train.py --prefer nvidia \
  --data X:\blackberry-splats\dataset --out opencl_out_nv \
  --data-factor 4 --steps 5000
```

Laptop with NVIDIA + Intel UHD: always `--prefer nvidia` (or the discrete
GPU name), never bare `--prefer intel`.

### 4) No OpenCL ICD (lab host, CI, CPU-only)

```bash
python engines_opencl/run_train.py --cpu-fallback \
  --data /usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset \
  --out opencl_out_cpu --data-factor 4 --steps 500
```

### Resume / interrupt

Same flags as the original run, plus `--resume` (settings must match `cfg.json`):

```bash
python engines_opencl/run_train.py --prefer arc --resume \
  --data X:\blackberry-splats\dataset --out opencl_out_mate \
  --data-factor 4 --steps 500
```

Ctrl-C writes `ckpt_latest.pt`.

## Checkpoints

- `ckpt_latest.pt` every `--ckpt-every` steps (default 50), also on Ctrl-C
- Final PLY: `splat_opencl.ply`

## Drivers

| GPU | What to install | `--prefer` hint |
|--|--|--|
| Intel Arc (MateBook) | Intel Graphics / Compute Runtime OpenCL ICD | `arc` or `intel` |
| Intel UHD only | same stack; expect weak iGPU | `uhd` / `intel` (slow) |
| AMD | ROCm or vendor OpenCL ICD | `amd` / `radeon` |
| NVIDIA | NVIDIA OpenCL ICD (or use `engines_cuda`) | `nvidia` |
| none | - | `--cpu-fallback` |

Verify:

```bash
python engines_opencl/list_devices.py
```

Then run **without** `--cpu-fallback` so the OpenCL device is bound for
preview paths.

## Layout

```
engines_opencl/
  cl/rasterize.cl          # project + atomic splat + finalize
  raster/cl_device.py
  raster/cl_rasterizer.py
  train/hybrid_raster.py   # OpenCL no-grad; CPU when needs_grad
  list_devices.py
  run_train.py
```
