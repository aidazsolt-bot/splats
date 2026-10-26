# splats

Macro Gaussian splat tooling for COLMAP + RGBA datasets (e.g. Blackberry).

## Pipelines

- [`engines/`](engines/) - CPU, deterministic, no training
- [`engines_cuda/`](engines_cuda/) - PyTorch + gsplat training (photorealistic; needs NVIDIA CUDA)
- [`colab/`](colab/) - Google Colab notebook (CUDA GPU in the cloud; same stack as `engines_cuda`)
- [`engines_cpu_train/`](engines_cpu_train/) - pure PyTorch training on CPU (quality-first; slow)
- [`engines_opencl/`](engines_opencl/) - OpenCL forward rasterizer (Intel Arc / AMD), hybrid train
- [`engines_povray/`](engines_povray/) - COLMAP cameras/images/mesh -> POV-Ray scene + renders

**Colab / PyTorch3D:** Colab has NVIDIA CUDA. Use **gsplat** via `engines_cuda` for 3DGS. PyTorch3D is optional (meshes only), not a splat trainer - see [`colab/README.md`](colab/README.md).

## Dataset

COLMAP text + RGBA PNGs, e.g. lab host:

```text
/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset/
  cameras.txt
  images.txt
  points3D.txt
  images/*.png
```

On Colab Drive: `MyDrive/blackberry-splats/dataset/` (same layout).

## Usage

### Google Colab (recommended without local CUDA)

1. Open [`colab/blackberry_train_colab.ipynb`](colab/blackberry_train_colab.ipynb) (GitHub "Open in Colab" or upload).
2. Runtime → **GPU**.
3. Mount Drive, run cells (smoke, then full train).
4. Download `splat_*.ply` from the output folder.

Details: [`colab/README.md`](colab/README.md)

### CUDA / Torch training (local GPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r engines_cuda/requirements.txt
python engines_cuda/run_train.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out" \
  --max-steps 30000
```

Match the Torch wheel CUDA tag (`cu121`, `cu124`, ...) to the driver/toolkit. Optional: `--data-factor 2`, `--smoke`, `--resume` (same settings + `--out`).

Details: [`engines_cuda/README.md`](engines_cuda/README.md)

### CPU deterministic (no training)

```bash
pip install -r engines/requirements.txt
python engines/run_blackberry.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_out"
```

Details: [`engines/README.md`](engines/README.md)

### CPU quality training (no CUDA)

```bash
pip install -r engines_cpu_train/requirements.txt
python engines_cpu_train/run_train.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cpu_train_out"
```

Smoke: `python engines_cpu_train/run_train.py --smoke`. Resume: add `--resume` with the same flags. Details: [`engines_cpu_train/README.md`](engines_cpu_train/README.md).

### OpenCL training (Intel Arc / non-CUDA GPUs)

```bash
pip install -r engines_opencl/requirements.txt
python engines_opencl/list_devices.py
python engines_opencl/run_train.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out opencl_out
# after abort: same command + --resume
```

Use `--cpu-fallback` if no OpenCL ICD is installed. Details: [`engines_opencl/README.md`](engines_opencl/README.md).

### POV-Ray scene from COLMAP

```bash
python engines_povray/colmap_to_pov.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out povray_out \
  --mesh engines_out/mesh.ply
python engines_povray/render_views.py --scene-dir povray_out --indices 0
```

Details: [`engines_povray/README.md`](engines_povray/README.md).
