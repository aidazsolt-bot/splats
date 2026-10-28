# Google Colab (`engines_cuda`)

Cloud GPU training with the **same** code as a local NVIDIA laptop.
Branch: **`feature/colab-train` only**.

| Goal | Tool |
|------|------|
| Train photoreal splats | **gsplat** via `engines_cuda/run_train.py` |
| Mesh from trained Gaussians | **`engines_cuda/export/mesh_from_ckpt.py`** (Open3D Poisson) |
| Mesh without training | `engines/run_blackberry.py` |

PyTorch3D is **not** used in this pipeline (optional third-party lib; ignore it).

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

CUDA Toolkit 12.1 + MSVC + ninja: see [`../WINDOWS_CUDA_SETUP.md`](../WINDOWS_CUDA_SETUP.md).

## Train recipe (Postshot-like candidate)

```text
--strategy mcmc --cap-max 200000 --max-steps 45000 --data-factor 1 --antialiased
```

| GPU | `--data-factor` | Notes |
|-----|----------------|-------|
| T4 / GTX 1650 4 GB | `2`, then try `1` | 200k + factor 1 may OOM |
| L4 / A100 | `1` | Closest to Dany-scale target |

`--resume`: continues from `ckpt_latest.pt` if present; otherwise starts fresh (with a hint).

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

Notebook / note default `OUT`: `…/cuda_out_mcmc/`.

## Mesh after train

```bash
python engines_cuda/export/mesh_from_ckpt.py \
  --ckpt "$OUT/ckpt_45000.pt" \
  --out "$OUT/mesh_from_splat.ply"
```

(Also in the notebook, section 6.)

## Limits

- Free Colab can disconnect — checkpoints on Drive + `--resume`.
- Needs real NVIDIA CUDA (Colab GPU or local Toolkit/`nvcc`); not Mesa / Intel-only.

More: [`../engines_cuda/README.md`](../engines_cuda/README.md), [`../engines_cuda/STRATEGY_PLAN.md`](../engines_cuda/STRATEGY_PLAN.md).
