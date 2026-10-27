# engines_cuda - photorealistic 3DGS training (PyTorch + gsplat)

**Windows `cl.exe` / Ninja setup:** [install and activate the build tools](../WINDOWS_CUDA_SETUP.md).

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
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r engines_cuda/requirements.txt
python engines_cuda/run_train.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset"
# Windows: --data X:\blackberry-splats\dataset --max-steps 30000
```

Match the Torch wheel CUDA tag (`cu121`, `cu124`, ...) to the driver/toolkit on that machine.

### Windows: gsplat CUDA extension build

Tested working setup (training ran to step 20000 with this toolchain):

| Component | Version |
|--|--|
| OS | Windows 11 Home, build 26200 |
| GPU | NVIDIA GeForce GTX 1650, 4 GB (compute capability 7.5) |
| NVIDIA driver | 621.01 (supports CUDA up to 13.5) |
| CUDA Toolkit (`nvcc`) | 12.1 (V12.1.66) |
| Visual Studio Build Tools | 2022, 17.14 |
| MSVC toolset / `cl.exe` | 14.44.35207 / 19.44.35229 (x64) |
| Python | 3.12.9 (x64) |
| PyTorch | 2.5.1+cu121 |
| gsplat | 1.5.3 (CUDA extension JIT-compiled, ~5 min first run) |
| Ninja | 1.13.2 |
| NumPy / Pillow | 1.26.4 / 12.1.1 |

MSVC 17.14 is newer than CUDA 12.1 officially supports; [win_build_env.py](win_build_env.py)
adds `-allow-unsupported-compiler` and `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`
to make this combination build.

If gsplat has no prebuilt CUDA extension, it compiles one on first use. This
requires Ninja (included in `requirements.txt`), the CUDA toolkit (`nvcc`), and
Visual Studio 2022 Build Tools with **Desktop development with C++** (MSVC x64
tools and Windows SDK). Run training from **Developer PowerShell for VS 2022**
or an **x64 Native Tools Command Prompt for VS 2022**.

For `RuntimeError: Ninja is required to load C++ extensions`, install Ninja
using the same Python interpreter that runs training:

```powershell
python -m pip install ninja
ninja --version
nvcc --version
cl
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

`ninja`, `nvcc`, and `cl` must be available on `PATH` in that terminal. If
`ninja` is not found after installation, add that Python installation's Scripts
directory to `PATH` (for a user install, locate it with
`python -m site --user-site` and replace the trailing `site-packages` with
`Scripts`). Match the installed CUDA toolkit to `torch.version.cuda`.

The initial `ImportError: cannot import name 'csrc' from 'gsplat'` triggers this
build fallback; the final build error is the one to resolve. Merely importing
gsplat does not verify that its CUDA extension can build or load.

Confirm CUDA before a long run:

```bash
python -c "import torch; assert torch.cuda.is_available(); import gsplat; print(torch.cuda.get_device_name(0))"
```

## Train Blackberry

### Required dataset files

`--data` must point to a COLMAP **text** export:

| File | Required | Used for |
|--|--|--|
| `cameras.txt` | yes | intrinsics; `SIMPLE_PINHOLE`, `SIMPLE_RADIAL`, `PINHOLE`, `OPENCV` |
| `images.txt` | yes | camera pose per image and image file name |
| `points3D.txt` | yes | initial Gaussians (position + color); must not be empty |
| `images/` | yes | the images named in `images.txt` |
| `rigs.txt`, `frames.txt` | no | ignored |

Notes:

- Binary COLMAP files (`*.bin`) are not read; export them as text first.
- RGBA images are recommended: the alpha channel is the subject mask for the
  loss. RGB images work, but then every pixel counts as foreground.
- Lens distortion parameters are read but not applied during rendering, so
  undistorted images (`PINHOLE` cameras) give the best result.
- Intrinsics are rescaled to the actual image resolution, so images may be
  smaller than the resolution stored in `cameras.txt`.
- Every 8th image (`--test-every`) is held out for evaluation.

Example layout: `/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset`.

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

Resume with the same settings and output directory:

```powershell
python engines_cuda/run_train.py --resume --data X:\blackberry-splats\dataset --out cuda_out --data-factor 4
```

### Image resolution and RAM cache

- `--data-factor N` (default `1`): downscale training images by `N`. `1` trains
  at the original resolution; `4` turns 2560x3840 into 640x960. Part of the
  resume settings, so a resumed run must use the same value.
- `--cache-images` (default off): decode and downscale all images once at
  startup and keep them in RAM. Without it, each training step reads one image
  from disk. Use it for slow (network) drives; at factor 1 it needs about
  `width * height * 4` bytes per image (~37 MiB for 2560x3840).
- `--cache-workers N` (default `8`): parallel loader threads for `--cache-images`.

Both cache options can be changed between resumed runs.

`--resume` starts a new run if no checkpoint exists. Otherwise it restores the
Gaussians, optimizer state, and training step, rejecting incompatible settings.
`ckpt_latest.pt` is saved every `--ckpt-every` steps (default 500), at numbered
save steps, and on Ctrl-C.
Checkpoint files are written to a temporary file in the same directory and then
atomically replaced, preserving the previous checkpoint if writing fails.
Replacement retries briefly on permission/sharing errors; if the file remains
locked, saving fails explicitly. Close programs holding the checkpoint open,
check folder permissions, then use `--resume` to restart from the last saved step.

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
