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
3. Run sections 1-2c: install, verify versions/basic CUDA, load gsplat, test forward/backward.
   On T4, 2b mounts Drive and loads the prebuilt `gsplat_cuda.so` under
   `MyDrive/blackberry-splats/gsplat-prebuilt/py312-torch215-cu134-512d366-sm75/`
   (skips the long compile). Other GPUs JIT the full gsplat default (can take a long time).
4. Section 3 remounts Drive for dataset/output paths; then smoke and full training.
5. Keep `ckpt_latest.pt` on Drive; set `RESUME = True` after disconnect.

For the failure history, verified five-minute build result, upstream references,
and feature restrictions, see the [gsplat build report](GSPLAT_BUILD_REPORT.md).

## Pinned local-stack comparison

The notebook uses the CUDA 13.4 / PyTorch 2.15 / gsplat 1.6.0 stack (same as
`engines_cuda/requirements.txt`, upstream `main` tip `512d366`). An older
CUDA 12.1 / gsplat 1.5.3 path is obsolete and no longer documented as default.

| Component | Notebook pin |
|-----------|--------------|
| Python | `3.12.15`, downloaded separately with uv |
| PyTorch | `2.15.0+cu134` from the official `cu134` index |
| CUDA compiler (`nvcc`) | `13.4.92` |
| NVIDIA runtime package set | `cuda-toolkit[cudart]==13.4.1` (required by PyTorch) |
| NVIDIA compiler packages | `nvidia-cuda-nvcc`, `nvidia-cuda-crt`, `nvidia-nvvm`: `13.4.92` |
| NVIDIA CCCL headers | `nvidia-cuda-cccl==13.3.4.3.1` |
| gsplat | `1.6.0`, source commit `512d366b67073d77ca099ede742683c165dfc23b` |
| Ninja | `1.13.2` |
| Parallel compiler jobs | `MAX_JOBS=10` |

These pins are shown at the top of the notebook and implemented in
[`runtime.py`](runtime.py). Installation and training use
an isolated environment at `/content/splats-cu134-env`; the notebook kernel's
preinstalled Python/PyTorch are not used for training. A bootstrap uv
installation at `/content/splats-bootstrap` downloads Python 3.12.15 without
replacing or restricting the notebook kernel's Python version. The environment
includes its own pip; package installation does not depend on the kernel's pip
support for cross-interpreter installs. Requirements are constrained so they cannot
silently replace the selected PyTorch/Ninja versions. The actual gsplat source
commit and `nvcc` version are checked before compiling.
gsplat is installed from `nerfstudio-project/gsplat` at the full 40-character
commit above. Compiler packages are pinned separately so installing nvcc
13.4.92 does not override PyTorch's CUDA runtime requirements.

**T4 / sm_75:** section 2b prefers the Drive prebuilt full `gsplat_cuda.so`
(under `MyDrive/blackberry-splats/gsplat-prebuilt/...-sm75/`) and injects it
before `import gsplat`, so training skips ninja. **Other GPUs:** JIT the same
full default (2DGS/3DGUT on, all `SupportedChannels`) into a `full-sm*` cache.
The old RGB-only pins (`BUILD_2DGS=0`, `BUILD_3DGUT=0`, `NUM_CHANNELS=3`) are
decommissioned.

Every diagnostic and training cell imports its dependencies and reloads the
runtime helper independently. The helper validates the interpreter/packages and
restores `CUDA_HOME`, library paths, target architecture, extension cache and
`MAX_JOBS=10` in the child process environment. Subprocess output is piped back
into notebook output explicitly, not left in Colab's server logs.
The toolkit's `include/cccl` directory is prepended to `CPATH` so the host C++
compiler can resolve `cuda/std/...` headers as well as nvcc. Existing `CPATH`
entries are preserved.

**Kernel restart on the same VM:** rerun the desired test cell directly. For
training, paths saved by section 3 are reloaded automatically; remount Drive
with section 3 if necessary. **New/replaced VM:** `/content` and its environment,
helper checkout and extension cache are gone; rerun sections 1 and 2, then the
tests and Drive mount. Missing setup produces an actionable error, not a
`NameError` or an attempt to use Colab's system Python.

Ten simultaneous compiler jobs can exceed a 12 GB host's available RAM.
This is the requested parallelism, with **no silent automatic throttling**.
An OOM kill can prevent the final resource report from being printed.

NVIDIA's compiler, runtime and CCCL wheels supply the build toolkit. The
notebook selects that toolkit explicitly; it does not install a driver or
replace Colab's system CUDA/Python. An incompatible Colab driver causes a
visible failure instead of falling back to another CUDA version.

### Reading the comparison

- **2a fails:** the problem occurs before gsplat (environment/driver/PyTorch).
- **2b fails:** on T4, check Drive prebuilt path / SHA; on other GPUs retain the
  final compiler error. JIT builds on the CPU; verbose Ninja plus
  `time`/`resource` reporting show elapsed time and peak RSS (not after an OOM
  kill). A cache hit on rerun is not a cold-build measurement.
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
The shared trainer requires PyTorch >=2.7. The notebook uses gsplat commit
`512d366` (same as `engines_cuda`). `BUILD_NO_CUDA=1` defers compilation until
2b; on T4 the prebuilt `.so` replaces that JIT.

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
