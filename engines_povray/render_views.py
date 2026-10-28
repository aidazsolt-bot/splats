#!/usr/bin/env python3
"""Render one or more COLMAP views with POV-Ray (if installed)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_LOCAL_POVRAY = (_REPO.parent / "povray.git" / "unix" / "povray").resolve()


def _resolve_povray(explicit: str | None) -> str:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return str(p)
        found = shutil.which(explicit)
        if found:
            return found
        raise SystemExit(f"povray not found: {explicit}")
    env = os.environ.get("POVRAY")
    if env and Path(env).is_file():
        return env
    if _DEFAULT_LOCAL_POVRAY.is_file():
        return str(_DEFAULT_LOCAL_POVRAY)
    found = shutil.which("povray")
    if found:
        return found
    raise SystemExit(
        "povray not found. Prefer local build:\n"
        f"  {_DEFAULT_LOCAL_POVRAY}\n"
        "Or set POVRAY=/path/to/povray / install system povray."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Render POV-Ray scene for selected camera indices")
    ap.add_argument("--scene-dir", type=Path, required=True, help="Directory with scene.pov")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--indices", type=str, default="0", help="Comma list or 'all'")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--antialias", type=float, default=0.3)
    ap.add_argument(
        "--povray",
        type=str,
        default=None,
        help="POV-Ray binary (default: ../povray.git/unix/povray, else $POVRAY, else PATH)",
    )
    ap.add_argument(
        "--file-gamma",
        type=str,
        default="sRGB",
        help="File_Gamma INI value (sRGB for display-referred PNG)",
    )
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    pov = scene_dir / "scene.pov"
    if not pov.is_file():
        raise SystemExit(f"Missing {pov} - run colmap_to_pov.py first")

    povray_bin = _resolve_povray(args.povray)
    print(f"Using povray: {povray_bin}", flush=True)

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
        out_png = (out_dir / f"cam_{i:03d}.png").resolve()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            povray_bin,
            f"+W{args.width}",
            f"+H{args.height}",
            f"+A{args.antialias}",
            "-D",
            f"File_Gamma={args.file_gamma}",
            f"Display_Gamma={args.file_gamma}",
            f"Declare=CAM_INDEX={i}",
            f"+O{out_png}",
            str(pov.name),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=str(scene_dir.resolve()), check=True)
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
