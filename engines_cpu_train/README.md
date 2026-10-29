# engines_cpu_train - photorealistic 3DGS on CPU (no CUDA)

Pure PyTorch differentiable Gaussian rasterizer + progressive multi-resolution
training. Aimed at **higher photometric quality** than the default CUDA shortcut
recipe (`engines_cuda` with `--data-factor 2` and 30k steps) by:

- progressive `data_factor` 8 -> 4 -> 2 (finer pixels late)
- optional **LPIPS** perceptual loss
- stronger alpha masking / subject focus
- more total optimization steps (~30k across stages, denser late stage)
- no VRAM cap forcing aggressive downscale

Wall-clock is still slower than CUDA, but the CPU rasterizer uses **batched
stencil scatter** (optional image tiles) instead of a Python loop over every
Gaussian. Benchmark:

```bash
python engines_cpu_train/bench_raster_cpu.py --n 4000 --w 160 --h 240
```

OpenCL/CPU train flags: `--cpu-method batched|loop`, `--cpu-batch-size`,
`--cpu-threads`, `--cpu-tile-size`.

## Usage

```bash
python -m venv ~/tmp/bn_cpu_train_venv
source ~/tmp/bn_cpu_train_venv/bin/activate
pip install -r engines_cpu_train/requirements.txt

# full quality schedule (long on CPU)
python engines_cpu_train/run_train.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cpu_train_out"
# Windows: --data X:\blackberry-splats\dataset

# short verification
python engines_cpu_train/run_train.py --smoke --out "$HOME/tmp/cpu_train_smoke"

# resume after interrupt (same --out / schedule flags)
python engines_cpu_train/run_train.py --resume \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cpu_train_out"
# Windows: --data X:\blackberry-splats\dataset
```

Outputs: `splat_final.ply`, per-stage PLY/ckpt, `ckpt_latest.pt`, `train_meta.json`.
`--resume` without a checkpoint: hint + fresh start. With checkpoint: settings must match `cfg.json`.
Ctrl-C also writes `ckpt_latest.pt`.

Reuses COLMAP+RGBA loader and PLY export from `engines_cuda/` (same formats).

## vs CUDA trainer

| | `engines_cuda` | `engines_cpu_train` |
|--|--|--|
| Rasterizer | gsplat CUDA | pure PyTorch |
| Hardware | NVIDIA GPU | CPU (29+ cores help a bit) |
| Default recipe | factor 2, 30k | progressive 8/4/2 + LPIPS |
| Speed | minutes-hours | hours-days |
| Goal | fast photoreal baseline | quality-first photoreal |

## Layout

```
engines_cpu_train/
  raster/diff_gauss_cpu.py
  train/trainer.py
  train/losses.py
  run_train.py
```
