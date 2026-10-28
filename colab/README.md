# Google Colab training

Colab runtimes usually have an **NVIDIA GPU + CUDA**. That matches `engines_cuda`
(gsplat + PyTorch), not the CPU/OpenCL paths.

Use branch **`feature/colab-train`** (includes MCMC / antialiased / resume from the
CUDA train line).

## PyTorch3D?

**Yes, Colab can install PyTorch3D**, but it is a *different* stack:

| Library | Role |
|---------|------|
| **gsplat** / `engines_cuda` | 3D Gaussian Splatting training (photoreal splats) |
| **PyTorch3D** | meshes, point clouds, soft rasterization - not a 3DGS trainer |

This folder's notebook uses **`engines_cuda` (gsplat)**. An optional cell shows a
minimal PyTorch3D install for mesh experiments.

## Quick start

1. Open [`blackberry_train_colab.ipynb`](blackberry_train_colab.ipynb) in Colab  
   (GitHub → Open in Colab, or upload the notebook).
2. Runtime → Change runtime type → **GPU** (T4 / L4 / A100).
3. Run cells: clone `feature/colab-train`, mount Drive, smoke, then MCMC train.
4. Download `splat_*.ply` / keep `ckpt_latest.pt` on Drive for `--resume`.

## Full train recipe (Postshot-like candidate)

Same as the local Windows note / `engines_cuda/STRATEGY_PLAN.md`:

```text
--strategy mcmc --cap-max 200000 --max-steps 45000 --data-factor 1 --antialiased
```

| GPU | `--data-factor` | Notes |
|-----|----------------|-------|
| T4 | `2` then try `1` | 200k + factor 1 may OOM |
| L4 / A100 | `1` | Closest to Dany-scale target |

`--resume` with no checkpoint: hint + fresh start. With checkpoint: settings must match `cfg.json`.

## Dataset on Drive

```text
MyDrive/blackberry-splats/dataset/
  cameras.txt
  images.txt
  points3D.txt
  images/*.png
```

Same layout as Linux `/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset` and Windows `X:\blackberry-splats\dataset`.

Notebook default `OUT`: `MyDrive/blackberry-splats/cuda_out_mcmc/`.

## Limits

- Free Colab sessions can disconnect; prefer Drive paths for checkpoints.
- gsplat must match the runtime's Torch/CUDA build (notebook installs for the session).
- Mesa / Intel OpenCL / Huawei MateBook iGPUs are **not** substitutes for this path.

Full local CUDA docs: [`../engines_cuda/README.md`](../engines_cuda/README.md).
