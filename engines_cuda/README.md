# engines_cuda - photorealistic 3DGS training (PyTorch + gsplat)

Learned Gaussian splats via gradient descent (same family as Postshot / Inria 3DGS),
with **alpha-masked** photometric loss for cut-out macro subjects (Blackberry RGBA).

Requires **NVIDIA CUDA**. Lab hosts without a GPU cannot run training locally.

**No local NVIDIA?** Use Google Colab: [`../colab/README.md`](../colab/README.md) and
[`../colab/blackberry_train_colab.ipynb`](../colab/blackberry_train_colab.ipynb)
(same `engines_cuda` + gsplat stack on a cloud T4/L4/A100).

PyTorch3D is unrelated to this trainer (optional mesh tools only).

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
# Linux lab
python engines_cuda/run_train.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" --max-steps 30000
# Windows: --data X:\blackberry-splats\dataset
```

Match the Torch wheel CUDA tag (`cu121`, `cu124`, ...) to the driver/toolkit on that machine.

Confirm CUDA before a long run:

```bash
python -c "import torch; assert torch.cuda.is_available(); import gsplat; print(torch.cuda.get_device_name(0))"
```

## Train Blackberry

Dataset layout (COLMAP text + `images/*.png` RGBA):
Linux `/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset`, Windows `X:\blackberry-splats\dataset`.

```bash
python engines_cuda/run_train.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out" \
  --data-factor 2 \
  --max-steps 30000
```

Smoke test on GPU:

```bash
python engines_cuda/run_train.py --smoke --data-factor 4 \
  --out "$HOME/tmp/gs_smoke"
```

Resume after interrupt (same flags + `--out`):

```bash
python engines_cuda/run_train.py --resume \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out" \
  --data-factor 2 --max-steps 30000
```

`ckpt_latest.pt` is written every `--ckpt-every` steps (default 500) and on Ctrl-C.
Settings must match `cfg.json` in `--out`.

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
