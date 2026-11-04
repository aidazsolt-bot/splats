# engines_cuda - photorealistic 3DGS / 2DGS training (PyTorch + gsplat)

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
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu134
python -m pip install -r engines_cuda/requirements.txt
BUILD_NO_CUDA=1 python -m pip install --no-build-isolation \
  "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@512d366b67073d77ca099ede742683c165dfc23b"
python -m pip install --no-build-isolation -r engines_cuda/requirements-pytorch3d.txt
python engines_cuda/run_train.py --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset"
# Windows: --data X:\blackberry-splats\dataset --max-steps 30000
```

The pinned gsplat source matches Colab: upstream
`nerfstudio-project/gsplat@512d366` (version 1.6.0). It requires PyTorch 2.7 or
newer. Use the `cu134` wheel with CUDA Toolkit 13.4. `BUILD_NO_CUDA=1` defers
extension compilation until the first gsplat CUDA call (Colab T4 can skip that
via a Drive prebuilt `.so`; see [`../colab/README.md`](../colab/README.md)).

On Windows PowerShell, install the pinned gsplat source with:

```powershell
$env:BUILD_NO_CUDA = "1"
python.exe -m pip install --no-build-isolation "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@512d366b67073d77ca099ede742683c165dfc23b"
Remove-Item Env:BUILD_NO_CUDA
```

The Windows trainer uses the **full** gsplat default build (2DGS/3DGUT enabled,
complete `SupportedChannels` list). First JIT on MSVC can be very slow; Colab T4
can skip it via the Drive prebuilt `.so`.

### Windows: gsplat CUDA extension build

**Current pin** (shared with Colab / Rocky Linux lab):

| Component | Version |
|--|--|
| OS | Windows 11 Home |
| GPU | NVIDIA GeForce GTX 1650, 4 GB (sm_75) |
| CUDA Toolkit (`nvcc`) | 13.4.92 |
| Visual Studio Build Tools | 2022 |
| MSVC toolset / `cl.exe` | **14.38** / 19.38 (pin via `vcvars_ver=14.38`) |
| Python | 3.12.15 (x64) |
| PyTorch | 2.15.0+cu134 |
| gsplat | 1.6.0 @ `512d366` (JIT; first build can take a long time on MSVC) |
| Ninja | 1.13.2 |

[win_build_env.py](win_build_env.py) calls `vcvars64.bat -vcvars_ver=14.38` so
nvcc sees `cl` 19.38 (bare Developer shells often pick 14.44, which Toolkit
13.4 may reject). The trainer also adds `include\cccl` to the process-local
MSVC `INCLUDE` path when present.

**Older note (superseded):** a GTX 1650 host previously trained with CUDA
Toolkit 12.1 + PyTorch 2.5.1+cu121 + gsplat 1.5.3 (~5 min first JIT). That is
no longer the documented default.

MSVC `std::isfinite` in device code (`Cameras.cuh`) remains a Windows-only
compile risk on some gsplat revisions; Linux/GCC maps it to
`__builtin_isfinite` and builds cleanly. Prefer matching the Colab commit
`512d366` when iterating on Windows.

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
On Windows, `win_build_env.py` defaults CUDA extension compilation to 2 jobs;
set `MAX_JOBS` explicitly to override it or reduce peak host memory use.

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

### Refinement strategies

New runs use gsplat's `DefaultStrategy` unless `--strategy` is specified.
Available strategies:

- `--strategy default`: screen-space-gradient cloning/splitting, low-opacity and
  oversized-splat pruning. Main options: `--refine-start` (500),
  `--refine-stop` (15000), `--refine-every` (100), `--grow-grad2d` (0.0002),
  `--grow-scale3d` (0.01), `--prune-scale3d` (0.1), `--prune-opa` (0.005),
  `--reset-every` (3000), `--pause-refine-after-reset` (0), `--absgrad`, and
  `--revised-opacity`. With `--absgrad`, the default `--grow-grad2d` threshold
  is 0.0008 instead of 0.0002.
- `--strategy mcmc`: gsplat's fixed-budget MCMC refinement; low-opacity splats
  are relocated and new splats are added up to `--cap-max` (default 200000).
  Other options include `--refine-stop` (25000), `--refine-every` (100),
  `--prune-opa` (0.005), and `--noise-lr` (500000). The cap limits additions;
  it does not reduce a checkpoint that already exceeds the cap.
- `--strategy legacy`: retain the repository's original custom refinement
  behavior.

`--opacity-reg` defaults to 0.01 for MCMC and 0 otherwise; `--scale-reg`
defaults to 0.01. `--antialiased` enables gsplat antialiased rasterization
(3DGS only). `--no-densify` disables all strategy refinement (including MCMC
relocation and noise). Strategy configuration and refinement state are
checkpointed.

### 2DGS / surfel training

Surface-oriented **2D Gaussian Splatting** via gsplat `rasterization_2dgs`
(same family as [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting)).
Disks replace ellipsoids; normal consistency + distortion losses encourage a
coherent surface (helpful before mesh extract).

```bash
python engines_cuda/run_train.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out_2dgs" \
  --2dgs --strategy default --data-factor 2 --max-steps 30000
# optional: --prune-opa 0.05  (gsplat 2DGS example default)
# smoke: --smoke --2dgs
```

- `--2dgs` switches the rasterizer; incompatible with `--antialiased`.
- Normal / distortion losses default **on** with `--2dgs`
  (`--no-normal-loss` / `--no-dist-loss` to disable).
- Weights / schedules: `--normal-lambda` (0.05), `--normal-start-iter` (7000),
  `--dist-lambda` (0.01), `--dist-start-iter` (3000). Smoke keeps those
  starts (does not pull them into 500 steps - that caused NaN blow-ups).
- Disk thickness: `scale_2` frozen at `1e-6`; exported PLY omits `scale_2` so
  Assimp tags `GaussianLayout=2dgs` (like supersplat_issue117).
- `DefaultStrategy` densifies on `gradient_2dgs` (not `means2d`).
- Official TSDF mesh extract from 2DGS renders is still roadmap; current
  `export/mesh_from_ckpt.py` (Open3D Poisson) still works on the PLY/ckpt.

Resuming a pre-strategy checkpoint with no `--strategy` flag automatically
selects `legacy`. New checkpoints require the same strategy and its relevant
parameters when resumed. Cache options may still be changed between runs.

Example:

```powershell
python engines_cuda/run_train.py --data X:\blackberry-splats\dataset --out cuda_mcmc `
  --data-factor 4 --strategy mcmc --cap-max 200000 --max-steps 45000
```

On the GTX 1650 with 4 GB VRAM, treat the MCMC cap as tunable: image resolution,
view count and rendering workspace also affect memory use. If CUDA runs out of
memory, stop and resume with the same output directory and a lower cap only if
the current Gaussian count is already below it; otherwise start a new run.

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

Default backend is **Open3D Poisson**. Optional **PyTorch3D** path voxelizes means and runs `cubify` (`--backend pytorch3d`; install via `requirements-pytorch3d.txt` with `--no-build-isolation`).

```bash
python export/mesh_from_ckpt.py \
  --ckpt "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out/ckpt_30000.pt" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out/mesh_from_splat.ply"

python export/mesh_from_ckpt.py \
  --ckpt "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out/ckpt_30000.pt" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out/mesh_from_splat_p3d.ply" \
  --backend pytorch3d --grid-size 128
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
