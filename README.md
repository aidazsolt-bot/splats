# splats

Macro Gaussian splat tooling for COLMAP + RGBA datasets (e.g. Blackberry).

## Branch

**Only `feature/colab-train`.** Use it for local NVIDIA (GTX 1650 etc.) and for Google Colab.

```bash
git checkout feature/colab-train && git pull
```

## Pipelines

| Path | What |
|------|------|
| [`engines_cuda/`](engines_cuda/) | Photoreal 3DGS train (PyTorch + **gsplat**, needs NVIDIA CUDA) |
| [`colab/`](colab/) | Same trainer in a Colab notebook |
| [`engines/`](engines/) | CPU deterministic splat + **Open3D mesh** (no training) |
| [`engines_cuda/export/mesh_from_ckpt.py`](engines_cuda/export/mesh_from_ckpt.py) | Mesh from trained Gaussian means (Open3D Poisson) |
| [`engines_cpu_train/`](engines_cpu_train/) | Pure PyTorch train on CPU (very slow) |
| [`engines_opencl/`](engines_opencl/) | OpenCL hybrid train (non-CUDA GPUs; backward mostly CPU) |
| [`engines_povray/`](engines_povray/) | COLMAP + mesh -> POV-Ray scene |

Windows CUDA (Toolkit 12.1 / MSVC / ninja): [`WINDOWS_CUDA_SETUP.md`](WINDOWS_CUDA_SETUP.md), [`engines_cuda/README.md`](engines_cuda/README.md).  
Colab details: [`colab/README.md`](colab/README.md).

## Dataset

```text
cameras.txt  images.txt  points3D.txt  images/*.png   # RGBA recommended
```

| Host | `--data` |
|------|----------|
| Windows | `X:\blackberry-splats\dataset` |
| Linux lab | `/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset` |
| Colab Drive | `MyDrive/blackberry-splats/dataset` |

## Usage

### Google Colab

1. Open [`colab/blackberry_train_colab.ipynb`](colab/blackberry_train_colab.ipynb) → Runtime **GPU**.
2. Mount Drive, run smoke, then MCMC full train.
3. Checkpoints under `MyDrive/blackberry-splats/cuda_out_mcmc/` (`RESUME = True` after disconnect).

### CUDA train (local laptop / lab)

Pin: Torch **`cu121`** + CUDA Toolkit **12.1** + MSVC + `ninja`.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r engines_cuda/requirements.txt

# Windows - MCMC / Postshot-like candidate
python .\engines_cuda\run_train.py --resume `
  --data X:\blackberry-splats\dataset `
  --out cuda_out_mcmc `
  --strategy mcmc --cap-max 200000 `
  --max-steps 45000 --data-factor 1 --antialiased

# Linux
python engines_cuda/run_train.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/cuda_out_mcmc" \
  --strategy mcmc --cap-max 200000 --max-steps 45000 --data-factor 1 --antialiased
```

GTX 1650 (4 GB): OOM → `--data-factor 2` and/or lower `--cap-max`. Optional: `--cache-images`, `--smoke`.

Mesh after train:

```bash
python engines_cuda/export/mesh_from_ckpt.py \
  --ckpt cuda_out_mcmc/ckpt_45000.pt \
  --out cuda_out_mcmc/mesh_from_splat.ply
```

### CPU deterministic (no training)

```bash
pip install -r engines/requirements.txt
python engines/run_blackberry.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_out"
```

### CPU quality train (no CUDA, slow)

```bash
pip install -r engines_cpu_train/requirements.txt
python engines_cpu_train/run_train.py --hires --resume \
  --data X:\blackberry-splats\dataset \
  --out X:\blackberry-splats\engines_cpu_train_out
```

### OpenCL hybrid

```bash
pip install -r engines_opencl/requirements.txt
python engines_opencl/list_devices.py
python engines_opencl/run_train.py --prefer nvidia --resume \
  --data X:\blackberry-splats\dataset --out opencl_out --data-factor 4 --steps 5000
```

### POV-Ray

```bash
python engines_povray/colmap_to_pov.py \
  --data X:\blackberry-splats\dataset \
  --out povray_out \
  --mesh engines_out/mesh.ply
python engines_povray/render_views.py --scene-dir povray_out --indices 0
```
