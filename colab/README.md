# Google Colab (`engines_cuda`)

Cloud GPU training with the **same** code as a local NVIDIA laptop.
Branch: **`feature/colab-train` only**.

| Goal | Tool |
|------|------|
| Train photoreal splats | **gsplat** via `engines_cuda/run_train.py` |
| Mesh from trained Gaussians | **`engines_cuda/export/mesh_from_ckpt.py`** (`--backend open3d` Poisson default, or `pytorch3d` cubify) |
| Mesh without training | `engines/run_blackberry.py` |

## Quick start (Colab)

1. Open [`blackberry_train_colab.ipynb`](blackberry_train_colab.ipynb) (GitHub → Open in Colab).
2. Runtime → **GPU** (T4 / L4 / A100).
3. Run cells: clone `feature/colab-train`, mount Drive, smoke, then MCMC train.
4. Keep `ckpt_latest.pt` on Drive; set `RESUME = True` after disconnect.

## Same on Windows laptop

```powershell
git checkout feature/colab-train
git pull
python .\engines_cuda\run_train.py --resume `
  --data X:\blackberry-splats\dataset `
  --out cuda_out_mcmc `
  --strategy mcmc --cap-max 200000 `
  --max-steps 45000 --data-factor 1 --antialiased
```

CUDA Toolkit matching the PyTorch CUDA runtime + MSVC + Ninja: see
[`../WINDOWS_CUDA_SETUP.md`](../WINDOWS_CUDA_SETUP.md).
The shared trainer now requires PyTorch >=2.7 and pins gsplat 1.6.0 to
upstream commit `512d366`. The Colab notebook installs that source with
`BUILD_NO_CUDA=1` and uses one CUDA build job to limit peak host memory.

## Train recipe

**Do not resume** the diverged run in `…/cuda_out_mcmc` (`splat_07000` means exploded).

Notebook default (stable):

```text
--strategy default --max-steps 30000 --data-factor 2 --antialiased
# OUT: MyDrive/blackberry-splats/cuda_out_stable
```

MCMC only with low noise (`5e5` diverged on this COLMAP scale):

```text
--strategy mcmc --cap-max 200000 --noise-lr 5e3 --max-steps 30000 --data-factor 2 --antialiased
```

| GPU | `--data-factor` | Notes |
|-----|----------------|-------|
| T4 / GTX 1650 4 GB | `2`, then try `1` | 200k + factor 1 may OOM |
| L4 / A100 | `1` | Closest to Dany-scale target |

`--resume`: only on a healthy `OUT` after disconnect. After first `splat_*.ply`, check AABB extent (~20-70, not 1e5).

## Dataset

```text
# Colab Drive
MyDrive/blackberry-splats/dataset/
  cameras.txt
  images.txt
  points3D.txt
  images/*.png

# Windows
X:\blackberry-splats\dataset\   (same layout)
```

Notebook default `OUT`: `…/cuda_out_stable/`.

## Mesh after train

```bash
# Default: Open3D Poisson
python engines_cuda/export/mesh_from_ckpt.py \
  --ckpt "$OUT/ckpt_45000.pt" \
  --out "$OUT/mesh_from_splat.ply"

# PyTorch3D voxel cubify (after requirements.txt; needs --no-build-isolation)
pip install --no-build-isolation -r engines_cuda/requirements-pytorch3d.txt
python engines_cuda/export/mesh_from_ckpt.py \
  --ckpt "$OUT/ckpt_45000.pt" \
  --out "$OUT/mesh_from_splat_p3d.ply" \
  --backend pytorch3d --grid-size 128
```

(Also in the notebook, section 6.)

## Limits

- Free Colab can disconnect — checkpoints on Drive + `--resume`.
- Needs real NVIDIA CUDA (Colab GPU or local Toolkit/`nvcc`); not Mesa / Intel-only.

More: [`../engines_cuda/README.md`](../engines_cuda/README.md), [`../engines_cuda/STRATEGY_PLAN.md`](../engines_cuda/STRATEGY_PLAN.md).
