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

### Supported stack (pin this - do not mix)

| Piece | Use this |
|-------|----------|
| PyTorch wheel | **`cu121`** (e.g. `2.5.1+cu121` works; newer Torch is fine **only** if still `+cu121`) |
| CUDA Toolkit (`nvcc`) | **12.1** (`...\CUDA\v12.1`) |
| MSVC | VS **2022** Build Tools + workload **VCTools** (Desktop C++) |
| gsplat extras | **`ninja`** on PATH; first import JIT-compiles CUDA kernels |

PyPI "latest" Torch may be newer (e.g. **2.14.x**) - that does **not** mean you should jump to
Toolkit 12.4 / `cu124`. Match **wheel tag ↔ Toolkit major.minor**. Newer NVIDIA **drivers**
(e.g. UMD 13.x) are OK with Toolkit 12.1.

Confirm CUDA before a long run:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
# gsplat must load the compiled extension:
python -c "from gsplat.cuda._backend import _C; assert _C is not None"
```

### Windows: CUDA Toolkit for gsplat (needed beyond the driver)

`nvidia-smi` + `torch.cuda.is_available() == True` is **not** enough. gsplat builds/loads
CUDA kernels and needs **`nvcc`** from the CUDA Toolkit. Symptom if missing:

`gsplat: No CUDA toolkit found. gsplat will be disabled` then `AttributeError: 'NoneType'...CameraModelType`.

For Torch **`cu121`**, install **CUDA Toolkit 12.1**. Prefer the **network installer** (~29 MB
bootstrap; you choose components) over the local/exe_local package (~3 GB full download):

[CUDA 12.1 network installer (Windows x86_64)](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network)

In the installer, choose **Custom** (not Express):

- Keep: **CUDA Development** (includes **nvcc**) and **CUDA Runtime**
- **Deselect Display Driver** if `nvidia-smi` already works - the toolkit often offers an
  older bundled driver; installing it can downgrade a newer Game Ready / Studio driver
- Skip: Demo Suite, Documentation, Nsight extras you do not need, samples

Notes from Windows installs (painful but real):

- The UI **install path on D:** does **not** keep C: small. NVIDIA still parks a large
  chunk on the system drive (Program Files, ProgramData, VS integration, redistributables).
  Reports: **~5–6 GB free needed on C:** even when "installing to D:" - free that space
  or the installer refuses / fails mid-way
- Prefer the network installer + Custom so you can **deselect Display Driver** and demos;
  the local ~3 GB package is worse, not better for disk
- If C: cannot spare ~6 GB: skip local Toolkit and use **Google Colab**
  ([`../colab/`](../colab/)) or another machine with room for `nvcc`

After install, **new** PowerShell (`nvcc` must be on PATH - Toolkit often lands under
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1` even if you picked D:):

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
$env:Path = "$env:CUDA_HOME\bin;" + $env:Path
nvcc --version

# gsplat JIT-builds CUDA kernels - needs ninja + MSVC (VS Build Tools "Desktop C++")
pip install ninja
pip uninstall gsplat -y
pip install gsplat --no-cache-dir

# First import may compile for a few minutes
python -c "from gsplat.cuda._backend import _C; print('gsplat CUDA ok', _C is not None)"

python .\engines_cuda\run_train.py --resume `
  --data X:\blackberry-splats\dataset --out cuda_out --data-factor 4
```

- `Ninja is required to load C++ extensions` -> `pip install ninja`, retry
- `cl.exe` / MSVC missing -> install Build Tools (winget, elevated / UAC OK):

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-package-agreements --accept-source-agreements --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

  Then **new** PowerShell (or "x64 Native Tools Command Prompt"), set `CUDA_HOME`/`Path` as above,
  and re-run the `from gsplat.cuda._backend import _C` check (first compile can take minutes).

`where.exe nvcc` empty while files exist under `...\CUDA\v12.1\bin` -> PATH/`CUDA_HOME` not set
(Toolkit install succeeded; shell just cannot see `nvcc`).

GTX 1650 (4 GB): start with `--data-factor 4` (or `8` on OOM).

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
`--resume` with no checkpoint: hint + fresh start. With checkpoint: settings must match `cfg.json`.

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
