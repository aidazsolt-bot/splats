# Google Colab training

Colab runtimes usually have an **NVIDIA GPU + CUDA**. That matches `engines_cuda`
(gsplat + PyTorch), not the CPU/OpenCL paths.

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
3. Run cells: clone `feature/colab-train`, mount Drive, smoke, then full train.
4. Download `splat_*.ply` (and optional `ckpt_*.pt`) from the output folder.

Suggested flags:

| GPU | `--data-factor` | Notes |
|-----|----------------|-------|
| T4 | `4` then try `2` | Safer VRAM for Blackberry macros |
| L4 / A100 | `2` or `1` | Higher quality, more VRAM |

Full local-CUDA docs: [`../engines_cuda/README.md`](../engines_cuda/README.md).

## Dataset on Drive

Upload or sync:

```text
MyDrive/blackberry-splats/dataset/
  cameras.txt
  images.txt
  points3D.txt
  images/*.png
```

Same layout as Linux `/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset` and Windows `X:\blackberry-splats\dataset`.

Outputs default under `MyDrive/blackberry-splats/engines_cuda_out/` (notebook `OUT`).

## Resume

After a disconnect, re-run the train cell with the same `DATA` / `OUT` / flags and `RESUME = True`
(`--resume`). Loads `ckpt_latest.pt` from Drive; settings must match `cfg.json`.

## Limits

- Free Colab sessions can disconnect; prefer Drive paths for checkpoints.
- gsplat must match the runtime's Torch/CUDA build (notebook installs for the session).
- Mesa / Intel OpenCL / Huawei MateBook iGPUs are **not** substitutes for this path.
