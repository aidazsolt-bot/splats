# splats

Macro Gaussian splat tooling for COLMAP + RGBA datasets (e.g. Blackberry).

## Branch

**Use only `feature/colab-train`.** Same code for:

- **Local NVIDIA laptop** (`engines_cuda` + gsplat / CUDA Toolkit) — e.g. GTX 1650
- **Google Colab** ([`colab/blackberry_train_colab.ipynb`](colab/blackberry_train_colab.ipynb))

```bash
git checkout feature/colab-train && git pull
```

## Pipelines

- [`engines/`](engines/) - CPU, deterministic, no training
- [`engines_cuda/`](engines_cuda/) - PyTorch + gsplat training (photorealistic; needs NVIDIA CUDA)
- [`colab/`](colab/) - Google Colab notebook (same `engines_cuda` stack)
- [`engines_cpu_train/`](engines_cpu_train/) - pure PyTorch training on CPU (quality-first; slow)
- [`engines_opencl/`](engines_opencl/) - OpenCL forward rasterizer (Intel Arc / AMD), hybrid train
- [`engines_povray/`](engines_povray/) - COLMAP cameras/images/mesh -> POV-Ray scene + renders

Windows CUDA setup (Toolkit / MSVC / ninja): [`WINDOWS_CUDA_SETUP.md`](WINDOWS_CUDA_SETUP.md) and [`engines_cuda/README.md`](engines_cuda/README.md).

## Usage

**Dataset `--data`:** Linux `/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset` · Windows `X:\blackberry-splats\dataset` · Colab `MyDrive/blackberry-splats/dataset`

### CUDA / Torch training (local NVIDIA laptop or lab GPU)

Same branch as Colab. Pin: Torch **`cu121`** + CUDA Toolkit **12.1** + MSVC + `ninja`.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r engines_cuda/requirements.txt

# Windows laptop (MCMC / Postshot-like candidate)
python .\engines_cuda\run_train.py --resume `
  --data X:\blackberry-splats\dataset `
  --out cuda_out_mcmc `
  --strategy mcmc --cap-max 200000 `
  --max-steps 45000 --data-factor 1 --antialiased

# Linux lab
python engines_cuda/run_train.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" --max-steps 30000
```

GTX 1650 (4 GB): if OOM, use `--data-factor 2` and/or lower `--cap-max`.
Optional: `--cache-images`, `--smoke`. Details: [`engines_cuda/README.md`](engines_cuda/README.md).

Required in `--data`: COLMAP `cameras.txt`, `images.txt`, `points3D.txt` plus `images/` (RGBA recommended). See [required dataset files](engines_cuda/README.md#required-dataset-files).

### CPU deterministic (no training)

```bash
pip install -r engines/requirements.txt
python engines/run_blackberry.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_out"
```

Details: [`engines/README.md`](engines/README.md)
