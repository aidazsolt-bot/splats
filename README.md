# splats

Macro Gaussian splat tooling for COLMAP + RGBA datasets (e.g. Blackberry).

## Pipelines

- [`engines/`](engines/) - CPU, deterministic, no training
- [`engines_cuda/`](engines_cuda/) - PyTorch + gsplat training (photorealistic; needs NVIDIA CUDA)
- [`colab/`](colab/) - Google Colab notebook (CUDA GPU in the cloud; same stack as `engines_cuda`)
- [`engines_cpu_train/`](engines_cpu_train/) - pure PyTorch training on CPU (quality-first; slow)
- [`engines_opencl/`](engines_opencl/) - OpenCL forward rasterizer (Intel Arc / AMD), hybrid train
- [`engines_povray/`](engines_povray/) - COLMAP cameras/images/mesh -> POV-Ray scene + renders

**Colab:** open [`colab/blackberry_train_colab.ipynb`](colab/blackberry_train_colab.ipynb) on branch `feature/colab-train` (MCMC recipe). Details: [`colab/README.md`](colab/README.md).

## Usage

**Dataset `--data`:** Linux `/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset` · Windows `X:\blackberry-splats\dataset` · Colab `MyDrive/blackberry-splats/dataset`

### CUDA / Torch training (GPU machine)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r engines_cuda/requirements.txt
python engines_cuda/run_train.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" --max-steps 30000
```

Match the Torch wheel CUDA tag (`cu121`, `cu124`, ...) to the driver/toolkit on that machine.
Tested Windows versions (driver, CUDA, MSVC, torch, gsplat): see [engines_cuda/README.md](engines_cuda/README.md#windows-gsplat-cuda-extension-build).

Optional flags: `--data-factor 2` (default `1` = original resolution), `--cache-images` (preload images into RAM), `--out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out"`, `--smoke` (short verification run).
`--cache-images` and `--cache-workers` (number of parallel threads that read/decode/downscale images during the one-time preload; default `8`) can be changed freely between resumed runs; `--data-factor` must stay the same.

Details: [`engines_cuda/README.md`](engines_cuda/README.md)

Required in `--data`: COLMAP text files `cameras.txt`, `images.txt`, `points3D.txt` plus the `images/` folder (RGBA recommended; alpha = subject mask). See [required dataset files](engines_cuda/README.md#required-dataset-files).

### CPU deterministic (no training)

```bash
pip install -r engines/requirements.txt
python engines/run_blackberry.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_out"
```

Details: [`engines/README.md`](engines/README.md)
