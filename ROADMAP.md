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

### 2DGS TSDF mesh extraction

Official **TSDF / unbounded mesh** export from 2DGS depth renders
(`hbb1/2d-gaussian-splatting` `render.py --mesh_res` / `--unbounded`).
Complements Open3D/Poisson on our ckpt PLY.

- **Train path:** done - `engines_cuda/run_train.py --2dgs` (gsplat
  `rasterization_2dgs` + normal/dist losses)
- **Still open:** port or wrap the official mesher into
  `engines_cuda/export/` (or call out to their `render.py`)

## Done / current defaults

- CUDA train + Colab: PyTorch `2.15.0+cu134`, gsplat `1.6.0` @ `512d366`
- Colab T4: Drive prebuilt `gsplat_cuda.so` (skip long JIT)
- Stable recipe docs; tiled CPU / OpenCL hybrid on this branch
- 2DGS / surfel train: `engines_cuda --2dgs` (normal + distortion regs)
