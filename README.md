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
| [`engines_cuda/`](engines_cuda/) | Photoreal 3DGS / **2DGS** train (PyTorch + **gsplat**, NVIDIA CUDA) |
| [`colab/`](colab/) | Same trainer as a Colab notebook |
| [`engines_cuda/export/mesh_from_ckpt.py`](engines_cuda/export/mesh_from_ckpt.py) | Mesh from trained Gaussians (Open3D Poisson or PyTorch3D cubify) |
| [`engines/`](engines/) | CPU deterministic splat + mesh (no training) |

Also in-tree (secondary):

- [`engines_opencl/`](engines_opencl/) - no-CUDA hosts (Huawei MateBook **Intel Arc**, AMD, NVIDIA OpenCL); GPU command examples in that README
- [`engines_cpu_train/`](engines_cpu_train/) - pure PyTorch CPU train
- [`engines_povray/`](engines_povray/) - POV-Ray export/render


Windows CUDA setup: [`WINDOWS_CUDA_SETUP.md`](WINDOWS_CUDA_SETUP.md) · [`engines_cuda/README.md`](engines_cuda/README.md)  
Colab: [`colab/README.md`](colab/README.md) · Roadmap: [`ROADMAP.md`](ROADMAP.md)

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

Open [`colab/blackberry_train_colab.ipynb`](colab/blackberry_train_colab.ipynb) → Runtime **GPU** (prefer T4) → sections 1-2c → Drive → smoke → train.  
On T4, 2b loads a Drive prebuilt `gsplat_cuda.so` (skips the long JIT).  
Default `OUT`: `MyDrive/blackberry-splats/cuda_out_stable/` · after disconnect: `RESUME = True`.  
Do **not** resume the diverged `cuda_out_mcmc` run.

### CUDA train (local laptop / lab)

Pin: Torch **`2.15.0+cu134`** + CUDA Toolkit **13.4** + gsplat **1.6.0** (`512d366`) + `ninja`.  
Windows also needs MSVC toolset **14.38** (see [`WINDOWS_CUDA_SETUP.md`](WINDOWS_CUDA_SETUP.md)).

```bash
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu134
pip install -r engines_cuda/requirements.txt
BUILD_NO_CUDA=1 pip install --no-build-isolation \
  "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@512d366b67073d77ca099ede742683c165dfc23b"
pip install --no-build-isolation -r engines_cuda/requirements-pytorch3d.txt

# Windows - stable recipe (or MCMC with low --noise-lr)
python .\engines_cuda\run_train.py `
  --data X:\blackberry-splats\dataset `
  --out cuda_out_stable `
  --strategy default --max-steps 30000 --data-factor 2 --antialiased

# 2DGS / surfel (surface-oriented; no --antialiased)
python .\engines_cuda\run_train.py `
  --data X:\blackberry-splats\dataset `
  --out cuda_out_2dgs `
  --2dgs --strategy default --max-steps 30000 --data-factor 2
```

GTX 1650 (4 GB): OOM → `--data-factor 2` and/or lower `--cap-max`.

Mesh after train:

```bash
python engines_cuda/export/mesh_from_ckpt.py \
  --ckpt cuda_out_stable/ckpt_30000.pt \
  --out cuda_out_stable/mesh_from_splat.ply
# optional: --backend pytorch3d --grid-size 128
```

### CPU deterministic (no training)

```bash
pip install -r engines/requirements.txt
python engines/run_blackberry.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_out"
```
