# Google Colab (`engines_cuda`)

Cloud GPU training with the **same** code as a local NVIDIA laptop.
Branch: **`feature/colab-train` only**.

| Goal | Tool |
|------|------|
| Train photoreal splats | **gsplat** via `engines_cuda/run_train.py` |
| Mesh from trained Gaussians | **`engines_cuda/export/mesh_from_ckpt.py`** (`--backend open3d` Poisson default, or `pytorch3d` cubify) |
| Mesh without training | `engines/run_blackberry.py` |

## Quick start (Colab)

1. Open [`blackberry_train_colab.ipynb`](blackberry_train_colab.ipynb) (GitHub → Open in Colab).
2. Start a fresh **GPU** runtime; prefer T4 for the local comparison.
3. Run sections 1-2c: install, verify versions/basic CUDA, build gsplat, test forward/backward.
4. Only after these pass: mount Drive, smoke, then full training.
5. Keep `ckpt_latest.pt` on Drive; set `RESUME = True` after disconnect.

## Pinned local-stack comparison

The notebook reproduces the currently installed local software versions, not
the older working CUDA 12.1 / gsplat 1.5.3 combination:

| Component | Notebook pin |
|-----------|--------------|
| Python | `3.12.9`, downloaded separately with uv |
| PyTorch | `2.15.0+cu134` from the official `cu134` index |
| CUDA compiler (`nvcc`) | `13.4.92` |
| NVIDIA toolkit Python package set | `cuda-toolkit[nvcc,cudart,cccl]==13.4.3` |
| gsplat | `1.6.0`, source commit `2a29ad06ba008bb34c9b942e82c2d46d39a194f4` |
| Ninja | `1.13.2` |
| Parallel compiler jobs | `MAX_JOBS=1` (local default: 10) |

These settings are at the top of the notebook. Installation and training use
an isolated environment at `/content/splats-cu134-env`; the notebook kernel's
preinstalled Python/PyTorch are not used for training. A bootstrap uv
installation at `/content/splats-bootstrap` downloads Python 3.12.9 without
replacing or restricting the notebook kernel's Python version. Requirements are constrained so they cannot
silently replace the selected PyTorch/Ninja versions. The actual gsplat source
commit and `nvcc` version are checked before compiling.

NVIDIA's compiler, runtime and CCCL wheels supply the build toolkit. The
notebook selects that toolkit explicitly; it does not install a driver or
replace Colab's system CUDA/Python. An incompatible Colab driver causes a
visible failure instead of falling back to another CUDA version.

### Reading the comparison

- **2a fails:** the problem occurs before gsplat (environment/driver/PyTorch).
- **2b fails:** retain the final compiler error. This phase builds on the CPU;
  verbose Ninja output plus `/usr/bin/time -v` show elapsed time and peak host
  RAM. A cache hit on rerun is not a cold-build measurement.
- **2c fails:** the extension built, but its tiny forward/backward GPU test
  failed.
- **2c passes, training fails:** investigate the dataset, training settings and
  memory use separately.

Colab uses **Linux/GCC**, not Windows/MSVC, and its driver and host RAM
can differ. A T4 shares compute capability 7.5 with the
local GTX 1650, but has more VRAM. Success on Colab helps isolate a
Windows/toolchain issue; it does not establish that the local hardware is
defective or provide an identical hardware benchmark.

PyTorch3D is excluded from installation/build diagnostics. Section 6 can
install it explicitly for optional cubify mesh export.

## Same on Windows laptop

```powershell
git checkout feature/colab-train
git pull
python .\engines_cuda\run_train.py --resume `
  --data X:\blackberry-splats\dataset `
  --out cuda_out_mcmc `
  --strategy mcmc --cap-max 200000 `
  --max-steps 45000 --data-factor 1 --antialiased
```

CUDA Toolkit matching the PyTorch CUDA runtime + MSVC + Ninja: see
[`../WINDOWS_CUDA_SETUP.md`](../WINDOWS_CUDA_SETUP.md).
The shared trainer requires PyTorch >=2.7. For this comparison, the notebook
uses the local gsplat commit listed above rather than the `512d366` pin in
the general setup instructions. `BUILD_NO_CUDA=1` defers compilation to the
explicit build diagnostic, with one CUDA build job to limit peak host memory.

## Train recipe

**Do not resume** the diverged run in `…/cuda_out_mcmc` (`splat_07000` means exploded).

Notebook default (stable):

```text
--strategy default --max-steps 30000 --data-factor 2 --antialiased
# OUT: MyDrive/blackberry-splats/cuda_out_stable
```

MCMC only with low noise (`5e5` diverged on this COLMAP scale):

```text
--strategy mcmc --cap-max 200000 --noise-lr 5e3 --max-steps 30000 --data-factor 2 --antialiased
```

| GPU | `--data-factor` | Notes |
|-----|----------------|-------|
| T4 / GTX 1650 4 GB | `2`, then try `1` | 200k + factor 1 may OOM |
| L4 / A100 | `1` | Closest to Dany-scale target |

`--resume`: only on a healthy `OUT` after disconnect. After first `splat_*.ply`, check AABB extent (~20-70, not 1e5).

## Dataset

```text
# Colab Drive
MyDrive/blackberry-splats/dataset/
  cameras.txt
  images.txt
  points3D.txt
  images/*.png

# Windows
X:\blackberry-splats\dataset\   (same layout)
```

Notebook default `OUT`: `…/cuda_out_stable/`.

## Mesh after train

```bash
# Default: Open3D Poisson
python engines_cuda/export/mesh_from_ckpt.py \
  --ckpt "$OUT/ckpt_45000.pt" \
  --out "$OUT/mesh_from_splat.ply"

# PyTorch3D voxel cubify (after requirements.txt; needs --no-build-isolation)
pip install --no-build-isolation -r engines_cuda/requirements-pytorch3d.txt
python engines_cuda/export/mesh_from_ckpt.py \
  --ckpt "$OUT/ckpt_45000.pt" \
  --out "$OUT/mesh_from_splat_p3d.ply" \
  --backend pytorch3d --grid-size 128
```

(Also in the notebook, section 6.)

## Limits

- Free Colab can disconnect — checkpoints on Drive + `--resume`.
- Needs real NVIDIA CUDA (Colab GPU or local Toolkit/`nvcc`); not Mesa / Intel-only.

More: [`../engines_cuda/README.md`](../engines_cuda/README.md), [`../engines_cuda/STRATEGY_PLAN.md`](../engines_cuda/STRATEGY_PLAN.md).
