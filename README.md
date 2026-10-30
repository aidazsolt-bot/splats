# splats

Macro Gaussian splat tooling for COLMAP + RGBA datasets (e.g. Blackberry).

## Branch

**Only `feature/colab-train`.** Local NVIDIA laptop and Google Colab use this same tree.

```bash
git checkout feature/colab-train && git pull
```

## Pipelines

| Path | What |
|------|------|
| [`engines_cuda/`](engines_cuda/) | Photoreal 3DGS train (PyTorch + **gsplat**, NVIDIA CUDA) |
| [`colab/`](colab/) | Same trainer as a Colab notebook |
| [`engines_cuda/export/mesh_from_ckpt.py`](engines_cuda/export/mesh_from_ckpt.py) | Mesh from trained Gaussians (Open3D Poisson or PyTorch3D cubify) |
| [`engines/`](engines/) | CPU deterministic splat + mesh (no training) |

Also in-tree (secondary):

- [`engines_opencl/`](engines_opencl/) - no-CUDA hosts (Huawei MateBook **Intel Arc**, AMD, NVIDIA OpenCL); GPU command examples in that README
- [`engines_cpu_train/`](engines_cpu_train/) - pure PyTorch CPU train
- [`engines_povray/`](engines_povray/) - POV-Ray export/render

Windows CUDA setup: [`WINDOWS_CUDA_SETUP.md`](WINDOWS_CUDA_SETUP.md) · [`engines_cuda/README.md`](engines_cuda/README.md)  
Colab: [`colab/README.md`](colab/README.md)

## Dataset

```text
cameras.txt  images.txt  points3D.txt  images/*.png
```

| Host | `--data` |
|------|----------|
| Windows | `X:\blackberry-splats\dataset` |
| Linux lab | `/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset` |
| Colab Drive | `MyDrive/blackberry-splats/dataset` |

## Usage

### Google Colab

Open [`colab/blackberry_train_colab.ipynb`](colab/blackberry_train_colab.ipynb) → Runtime **GPU** → mount Drive → smoke → MCMC train.  
`OUT`: `MyDrive/blackberry-splats/cuda_out_mcmc/` · after disconnect: `RESUME = True`.

### CUDA train (local laptop / lab)

Pin: Torch **`cu121`** + CUDA Toolkit **12.1** + MSVC + `ninja`.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r engines_cuda/requirements.txt
pip install --no-build-isolation -r engines_cuda/requirements-pytorch3d.txt

# Windows - MCMC / Postshot-like candidate
python .\engines_cuda\run_train.py --resume `
  --data X:\blackberry-splats\dataset `
  --out cuda_out_mcmc `
  --strategy mcmc --cap-max 200000 `
  --max-steps 45000 --data-factor 1 --antialiased
```

GTX 1650 (4 GB): OOM → `--data-factor 2` and/or lower `--cap-max`.

Mesh after train:

```bash
python engines_cuda/export/mesh_from_ckpt.py \
  --ckpt cuda_out_mcmc/ckpt_45000.pt \
  --out cuda_out_mcmc/mesh_from_splat.ply
# optional: --backend pytorch3d --grid-size 128
```

### CPU deterministic (no training)

```bash
pip install -r engines/requirements.txt
python engines/run_blackberry.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_out"
```
