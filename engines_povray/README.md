# engines_povray - COLMAP cameras + geometry -> POV-Ray scene

Builds a POV-Ray 3.7 scene from:

- COLMAP `cameras.txt` / `images.txt` / `points3D.txt`
- RGBA capture images (optional textured billboards)
- optional reconstructed mesh (`engines_out/mesh.ply`) with vertex colors

## Usage

```bash
# 1) Generate scene (mesh if engines_out/mesh.ply exists, else point spheres)
python engines_povray/colmap_to_pov.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out povray_out \
  --mesh engines_out/mesh.ply

# optional photo planes
python engines_povray/colmap_to_pov.py \
  --data "/usr/src/TestAdnroidNDKSDK/blackberry-splats/dataset" \
  --out povray_out \
  --mesh engines_out/mesh.ply \
  --billboards --billboard-every 8

# 2) Render (requires povray on PATH)
python engines_povray/render_views.py \
  --scene-dir povray_out \
  --indices 0,10,20 \
  --width 1280 --height 1920
```

Manual render:

```bash
cd povray_out
povray +W1280 +H1920 +A0.3 -D Declare=CAM_INDEX=0 scene.pov
```

## Install POV-Ray

Not bundled. On a machine with package admin:

- Fedora/RHEL: `dnf install povray`
- Debian/Ubuntu: `apt install povray`
- Or https://www.povray.org/download/

## What gets written

| File | Role |
|------|------|
| `scene.pov` | Master scene, lights, `#include`s |
| `cameras.inc` | One macro per COLMAP view (`Cam_i`, `UseCamera`) |
| `geometry_mesh.inc` | `mesh2` with vertex-color textures |
| `geometry_points.inc` | Sparse point spheres |
| `billboards.inc` + `textures/` | Optional image planes |
| `cameras.json` | Index -> image name map |

Cameras are converted from COLMAP/OpenCV (look +Z, Y down) to POV `location` / `look_at` / `sky` / horizontal `angle`. Geometry is wrapped in `scale <1,-1,-1>` for a Y-up-friendly POV viewport.
