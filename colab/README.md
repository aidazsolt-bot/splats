# Google Colab training

Colab runtimes usually have an **NVIDIA GPU + CUDA**. That matches `engines_cuda`
(gsplat + PyTorch), not the CPU/OpenCL paths.

## PyTorch3D?

**Yes, Colab can install PyTorch3D**, but it is a *different* stack:

| Library | Role |
|---------|------|
| **gsplat** / `engines_cuda` | 3D Gaussian Splatting training (what we want for photoreal splats) |
| **PyTorch3D** | meshes, point clouds, soft rasterization, silhouette losses – useful for mesh/refine, not a drop-in 3DGS trainer |

This folder’s notebook uses **`engines_cuda` (gsplat)**. An optional cell shows a
minimal PyTorch3D install for mesh experiments.

## Quick start

1. Open [`blackberry_train_colab.ipynb`](blackberry_train_colab.ipynb) in Colab  
   (GitHub → Open in Colab, or upload the notebook).
2. Runtime → Change runtime type → **GPU** (T4/L4/A100).
3. Run all cells. Point `--data` at your Drive copy of the Blackberry COLMAP dataset.
4. Download `splat_*.ply` from the output folder.

## Dataset on Drive

Upload or sync:

```text
MyDrive/blackberry-splats/dataset/
  cameras.txt
  images.txt
  points3D.txt
  images/*.png
```

Same layout as `/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset` on the lab host.
