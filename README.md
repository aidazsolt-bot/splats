# splats

Macro Gaussian splat tooling for COLMAP + RGBA datasets (e.g. Blackberry).

## Pipelines

- [`engines/`](engines/) - CPU, deterministic, no training
- [`engines_cuda/`](engines_cuda/) - PyTorch + gsplat training (photorealistic; needs NVIDIA CUDA)

## Usage

**Dataset `--data`:** Linux `/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset` · Windows `X:\blackberry-splats\dataset` · Colab `MyDrive/blackberry-splats/dataset`

### CUDA / Torch training (GPU machine)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r engines_cuda/requirements.txt
python engines_cuda/run_train.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" --max-steps 30000
```

Match the Torch wheel CUDA tag (`cu121`, `cu124`, ...) to the driver/toolkit on that machine.

Optional flags: `--data-factor 2` (default `1` = original resolution), `--cache-images` (preload images into RAM), `--out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out"`, `--smoke` (short verification run).
`--cache-images` and `--cache-workers` (number of parallel threads that read/decode/downscale images during the one-time preload; default `8`) can be changed freely between resumed runs; `--data-factor` must stay the same.

Details: [`engines_cuda/README.md`](engines_cuda/README.md)

### CPU deterministic (no training)

```bash
pip install -r engines/requirements.txt
python engines/run_blackberry.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_out"
```

Details: [`engines/README.md`](engines/README.md)
