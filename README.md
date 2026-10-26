# splats

Macro Gaussian splat tooling for COLMAP + RGBA datasets (e.g. Blackberry).

## Which path?

| Goal | Path | Needs |
|------|------|--------|
| Photoreal splat, no local NVIDIA | [`colab/`](colab/) | Google Colab GPU |
| Photoreal splat on NVIDIA box | [`engines_cuda/`](engines_cuda/) | CUDA + gsplat |
| Fast geometric PLY + mesh | [`engines/`](engines/) | CPU only |
| Train without CUDA (slow) | `engines_cpu_train/` | CPU, days+ |
| Non-NVIDIA GPU experiment | `engines_opencl/` | OpenCL ICD |
| Classic raytrace preview | `engines_povray/` | POV-Ray |

**Colab note:** Runtimes have NVIDIA CUDA. Training uses **gsplat / `engines_cuda`**, not PyTorch3D. PyTorch3D can be installed for mesh experiments but is not a 3DGS trainer. Details: [`colab/README.md`](colab/README.md).

Sibling feature branches that still carry the extra trees (merge later if needed):

- `feature/cpu-train-quality` - `engines_cpu_train/`
- `feature/opencl-train` - `engines_opencl/` (+ cpu/povray)
- `feature/povray-scene` - `engines_povray/`

This branch (`feature/colab-train`) focuses on Colab + `engines_cuda` + deterministic `engines/`.

## Dataset

COLMAP text + RGBA PNGs, e.g. lab host:

```text
/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset/
  cameras.txt
  images.txt
  points3D.txt
  images/*.png
```

On Colab Drive: `MyDrive/blackberry-splats/dataset/` (same layout).

## Usage

### Google Colab (recommended without local CUDA)

1. Open [`colab/blackberry_train_colab.ipynb`](colab/blackberry_train_colab.ipynb) (GitHub "Open in Colab" or upload).
2. Runtime → **GPU**.
3. Mount Drive, run cells (smoke, then full train).
4. Download `splat_*.ply` from the output folder.

Details: [`colab/README.md`](colab/README.md)

### CUDA / Torch training (local GPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r engines_cuda/requirements.txt
python engines_cuda/run_train.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_cuda_out" \
  --max-steps 30000
```

Match the Torch wheel CUDA tag (`cu121`, `cu124`, ...) to the driver/toolkit. Optional: `--data-factor 2`, `--smoke`.

Details: [`engines_cuda/README.md`](engines_cuda/README.md)

### CPU deterministic (no training)

```bash
pip install -r engines/requirements.txt
python engines/run_blackberry.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_out"
```

Details: [`engines/README.md`](engines/README.md)

### Other pipelines (other branches)

```bash
# CPU quality train - feature/cpu-train-quality
python engines_cpu_train/run_train.py --data ... --out ...

# OpenCL hybrid - feature/opencl-train
python engines_opencl/list_devices.py
python engines_opencl/run_train.py --data ... --out opencl_out

# POV-Ray - feature/povray-scene
python engines_povray/colmap_to_pov.py --data ... --out povray_out --mesh engines_out/mesh.ply
python engines_povray/render_views.py --scene-dir povray_out --indices 0
```
