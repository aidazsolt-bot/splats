# engines_opencl - 3DGS via OpenCL (Intel Arc / AMD / …)

OpenCL forward rasterizer for machines **without NVIDIA CUDA** (e.g. Huawei MateBook
with **Intel Arc**). Training uses a **hybrid** path:

- **Forward:** OpenCL kernels (`cl/rasterize.cl`) on the GPU/iGPU  
- **Backward:** pure PyTorch CPU rasterizer (same math as `engines_cpu_train`)

So you get OpenCL acceleration where ICD drivers exist, without requiring CUDA/gsplat.

## Usage

```bash
pip install -r engines_opencl/requirements.txt

# list devices (Intel Arc should appear if compute runtime is installed)
python engines_opencl/list_devices.py

# smoke (needs OpenCL device) or --cpu-fallback
python engines_opencl/run_train.py --smoke --out opencl_out --cpu-fallback

# Windows / dual GPU: force NVIDIA (not Intel UHD)
python engines_opencl/run_train.py --prefer nvidia \
  --data X:\blackberry-splats\dataset --out opencl_out --data-factor 4 --steps 5000

# after interrupt: same flags + --resume
python engines_opencl/run_train.py --prefer nvidia --resume \
  --data X:\blackberry-splats\dataset --out opencl_out --data-factor 4 --steps 5000

# Linux lab
python engines_opencl/run_train.py --prefer nvidia \
  --data /usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset \
  --out opencl_out --data-factor 4 --steps 5000
```

`--data` Windows: `X:\blackberry-splats\dataset` · Linux: `/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset`

Checkpoints: `ckpt_latest.pt` every `--ckpt-every` steps (default 50), also on Ctrl-C.
`--resume` continues if a checkpoint exists; otherwise prints a hint and starts fresh.
If a checkpoint exists, settings must match `cfg.json` in `--out`.

Training uses the **CPU differentiable** rasterizer (needed for autograd). OpenCL device
selection matters for ICD probing / preview. If Intel UHD appears despite index 1,
pull this fix and use `--prefer nvidia` (binds the NVIDIA platform explicitly).

**NVIDIA driver too old for PyTorch CUDA:** update the driver, or keep using this
OpenCL/CPU path (no CUDA required for training here).

## Drivers

This host may report `PLATFORM_NOT_FOUND` (no ICD). On a MateBook with Arc:

1. Install **Intel Graphics Compute Runtime** / oneAPI OpenCL ICD  
2. `python engines_opencl/list_devices.py` should list the Arc GPU  
3. Then run without `--cpu-fallback`

AMD: ROCm OpenCL. NVIDIA: OpenCL ICD works but prefer `engines_cuda` for CUDA.

## Layout

```
engines_opencl/
  cl/rasterize.cl          # project + atomic splat + finalize
  raster/cl_device.py
  raster/cl_rasterizer.py
  train/hybrid_raster.py   # torch.autograd.Function
  list_devices.py
  run_train.py
```
