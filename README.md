# splats

Macro Gaussian splat tooling for COLMAP + RGBA datasets (e.g. Blackberry).

## Pipelines

- [`engines/`](engines/) - CPU, deterministic, no training
- [`engines_cuda/`](engines_cuda/) - PyTorch + gsplat training (needs NVIDIA CUDA)
- [`engines_cpu_train/`](engines_cpu_train/) - pure PyTorch training on CPU (quality-first schedule)
- [`engines_povray/`](engines_povray/) - COLMAP cameras/images/mesh -> POV-Ray scene + renders
- [`engines_opencl/`](engines_opencl/) - OpenCL forward rasterizer (Intel Arc / AMD), hybrid train

## Usage

### CPU quality training (no CUDA)

```bash
pip install -r engines_cpu_train/requirements.txt
python engines_cpu_train/run_train.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cpu_train_out"
```

Smoke test: `python engines_cpu_train/run_train.py --smoke`

Details: [`engines_cpu_train/README.md`](engines_cpu_train/README.md)

### CUDA / Torch training (GPU machine)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r engines_cuda/requirements.txt
python engines_cuda/run_train.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" --max-steps 30000
```

Match the Torch wheel CUDA tag (`cu121`, `cu124`, ...) to the driver/toolkit on that machine.

Optional flags: `--data-factor 2`, `--out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out"`, `--smoke` (short verification run).

Details: [`engines_cuda/README.md`](engines_cuda/README.md)

### CPU deterministic (no training)

```bash
pip install -r engines/requirements.txt
python engines/run_blackberry.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_out"
```

Details: [`engines/README.md`](engines/README.md)

### OpenCL training (Intel Arc / non-CUDA GPUs)

```bash
pip install -r engines_opencl/requirements.txt
python engines_opencl/list_devices.py
python engines_opencl/run_train.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" --out opencl_out
```

Use `--cpu-fallback` if no OpenCL ICD is installed. Details: [`engines_opencl/README.md`](engines_opencl/README.md)

### POV-Ray scene from COLMAP

```bash
python engines_povray/colmap_to_pov.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out povray_out \
  --mesh engines_out/mesh.ply
python engines_povray/render_views.py --scene-dir povray_out --indices 0
```

Details: [`engines_povray/README.md`](engines_povray/README.md)
