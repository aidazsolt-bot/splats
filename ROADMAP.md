# Roadmap (`feature/colab-train`)

Living list. Not a commitment to dates.

## Planned

### Instruct-GS2GS text-edit pipe

Instruction-based editing of trained 3DGS scenes via InstructPix2Pix
([nerfstudio docs](https://docs.nerf.studio/nerfology/methods/igs2gs.html),
[`cvachha/instruct-gs2gs`](https://github.com/cvachha/instruct-gs2gs)).

- **Stack:** Python / Nerfstudio method `ns-train igs2gs` (gsplat under the hood)
- **VRAM:** ~15 GB (paper full model) - Colab L4/A100 class; not GTX 1650
- **Likely shape:** new `engines_igs2gs/` (or `colab/` cell) wrapping Nerfstudio,
  not a drop-in into `engines_cuda/run_train.py`
- **Open questions:** export/convert our `ckpt_*.pt` <-> Nerfstudio
  `nerfstudio_models`, or reimplement iterative dataset update on our trainer
- **Deps:** diffusion (InstructPix2Pix), full Nerfstudio install

### 2DGS + mesh extraction

Surface-oriented **2D Gaussian Splatting** with TSDF mesh export (bounded /
unbounded). Complements our current Open3D/Poisson mesh-from-3DGS path with a
geometry-first splat representation.

- **Project / paper:** [surfsplatting.github.io](https://surfsplatting.github.io/)
- **Official code:** [`hbb1/2d-gaussian-splatting`](https://github.com/hbb1/2d-gaussian-splatting)
  (`train.py` then `render.py ... --mesh_res` / `--unbounded`)
- **gsplat side:** 2DGS rasterization lives in our pinned gsplat (`rasterization_2dgs`);
  full mesh extract in-tree was historically incomplete - upstream issue/PR
  discussion points back to the official 2DGS `render.py` mesher
- **Likely shape:** `engines_2dgs/` or extend `engines_cuda/export/` after a 2DGS
  train path; COLMAP input already matches
- **Note:** not the same as Instruct-GS2GS (edit) or Nerfstudio Splatfacto
  (no mesh export there)

## Done / current defaults

- CUDA train + Colab: PyTorch `2.15.0+cu134`, gsplat `1.6.0` @ `512d366`
- Colab T4: Drive prebuilt `gsplat_cuda.so` (skip long JIT)
- Stable recipe docs; tiled CPU / OpenCL hybrid on this branch
