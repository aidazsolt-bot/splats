# engines_cuda - photorealistic 3DGS training (PyTorch + gsplat)

Learned Gaussian splats via gradient descent (same family as Postshot / Inria 3DGS),
with **alpha-masked** photometric loss for cut-out macro subjects (Blackberry RGBA).

Requires **NVIDIA CUDA**. This host without a GPU cannot run training; use a CUDA box.

## Why this vs `engines/` (no-train)

| | `engines/` | `engines_cuda/` |
|--|--|--|
| Method | closed-form PCA + bake | Adam + differentiable rasterization |
| Runtime | CPU minutes | GPU minutes..hours |
| Quality | geometric / textured mesh | photorealistic splat (SH view-dependence) |

## Usage

On a GPU machine:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r engines_cuda/requirements.txt
python engines_cuda/run_train.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" --max-steps 30000
```

Match the Torch wheel CUDA tag (`cu121`, `cu124`, ...) to the driver/toolkit on that machine.

Confirm CUDA before a long run:

```bash
python -c "import torch; assert torch.cuda.is_available(); import gsplat; print(torch.cuda.get_device_name(0))"
```

## Train Blackberry

Dataset layout (COLMAP text + `images/*.png` RGBA), e.g. `/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset`.

```bash
python engines_cuda/run_train.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out" \
  --data-factor 2 \
  --max-steps 30000
```

Smoke test on GPU:

```bash
python run_train.py --smoke --data-factor 4 --out /tmp/gs_smoke
```

Outputs:

- `splat_07000.ply`, `splat_15000.ply`, `splat_30000.ply` - viewer-compatible
- `ckpt_*.pt` - full torch checkpoint
- `cfg.json`, `train_meta.json`

## Mesh from trained means

```bash
python export/mesh_from_ckpt.py \
  --ckpt "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out/ckpt_30000.pt" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out/mesh_from_splat.ply"
```

## Training details

- Init: COLMAP `points3D` means + colors, knn scales
- Rasterizer: `gsplat.rendering.rasterization` (CUDA)
- Loss: masked L1 + SSIM on subject alpha, plus alpha BCE; optional random background
- Densify/prune: classic 3DGS clone/split/opacity-reset (CPU-side bookkeeping)
- SH degree ramps every 1000 steps up to 3

## Layout

```
engines_cuda/
  dataset/colmap_rgba.py   # COLMAP+RGBA loader (reuses engines/prep)
  train/trainer.py         # main loop
  train/losses.py
  export/ply.py
  export/mesh_from_ckpt.py
  run_train.py
```
