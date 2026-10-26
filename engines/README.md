# Deterministic splat + mesh engines (CPU, no training)

Builds a 3DGS-compatible `.ply` and a Poisson mesh from COLMAP poses + RGBA images
using closed-form geometry and photometry. No Adam / Postshot / densify-train loop.

## Pipeline

1. **Prep** - Scale COLMAP intrinsics to the actual PNG size (Blackberry: 5504x8256 -> 2560x3840).
2. **Geom** - Keep sparse `points3D` that project into alpha masks; densify by kNN midpoints; optional ORB stereo.
3. **Splat** - Local PCA -> anisotropic scales/rotation; multi-view weighted RGB; opacity from visibility*alpha.
4. **Mesh** - Open3D Poisson + multi-view vertex color bake.

## Math (short)

- Covariance at point `i`: `C_i = (1/(k-1)) sum_j (x_j - x_i)(x_j - x_i)^T` over kNN.
- Scales = `sqrt(eigenvalues(C_i))`, rotation = eigenvectors as quaternion.
- Color: `rgb = sum_v w_v * I_v(pi_v(x)) / sum_v w_v` with `w_v = 1_{z>0} * alpha * cos^2`.
- Opacity: `o = 1 - exp(-beta * mean_alpha * log(1 + n_vis))`, stored as logit in the PLY.
- SH DC: `f_dc = (rgb - 0.5) / 0.28209479177387814` (standard 3DGS).

## Run

```bash
# venv used on this host
source "$HOME/tmp/bn_engines_venv/bin/activate"
cd /usr/src/TestAdnroidNDKSDK/danybitte.splats.git/engines
python run_blackberry.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out "/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_out" \
  --downscale 4 \
  --densify-rounds 1
# Windows: --data X:\blackberry-splats\dataset
```

Outputs under `/usr/src/TestAdnroidNDKSDK/blackberry-splats/engines_out/`:

- `colmap_scaled/` - intrinsics matched to PNG resolution
- `dense_cloud.ply` - working point cloud
- `deterministic_splat.ply` - Gaussian splat (viewer-compatible layout)
- `mesh.obj` / `mesh.ply` - Poisson mesh with vertex colors
- `run_meta.json` - counts and timing

## Limits

- Not a substitute for trained Postshot quality on fuzzy/shiny macros.
- Full-resolution multi-view without `--downscale` is slow on CPU.
- Mesa/OpenGL does not accelerate this path.

## Modules

| Path | Role |
|------|------|
| `prep/colmap_io.py` | COLMAP text R/W + intrinsic scale |
| `prep/dataset.py` | Dataset loader + RGBA |
| `geom/dense_cpu.py` | Mask filter, densify, optional stereo |
| `splat/build_gaussians.py` | PCA Gaussians + PLY export |
| `mesh/build_mesh.py` | Poisson + color bake |
| `run_blackberry.py` | CLI |
