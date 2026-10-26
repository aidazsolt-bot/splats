#!/usr/bin/env python3
"""Render one or more COLMAP views with POV-Ray (if installed)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Render POV-Ray scene for selected camera indices")
    ap.add_argument("--scene-dir", type=Path, required=True, help="Directory with scene.pov")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--indices", type=str, default="0", help="Comma list or 'all'")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--antialias", type=float, default=0.3)
    ap.add_argument("--povray", type=str, default="povray")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    pov = scene_dir / "scene.pov"
    if not pov.is_file():
        raise SystemExit(f"Missing {pov} - run colmap_to_pov.py first")

    povray_bin = shutil.which(args.povray)
    if not povray_bin:
        raise SystemExit(
            "povray not found on PATH. Install POV-Ray 3.7+ then re-run.\n"
            "  Fedora/RHEL:  (admin) dnf install povray\n"
            "  Debian/Ubuntu: (admin) apt install povray\n"
            "  Or download: https://www.povray.org/download/"
        )

    cams_path = scene_dir / "cameras.json"
    n = 1
    if cams_path.is_file():
        cams = json.loads(cams_path.read_text(encoding="utf-8"))
        n = len(cams)
    if args.indices.strip().lower() == "all":
        indices = list(range(n))
    else:
        indices = [int(x) for x in args.indices.split(",") if x.strip() != ""]

    out_dir = Path(args.out_dir) if args.out_dir else scene_dir / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in indices:
        out_png = out_dir / f"cam_{i:03d}.png"
        cmd = [
            povray_bin,
            f"+W{args.width}",
            f"+H{args.height}",
            f"+A{args.antialias}",
            "-D",
            f"Declare=CAM_INDEX={i}",
            f"+O{out_png}",
            str(pov.name),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=str(scene_dir), check=True)
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
