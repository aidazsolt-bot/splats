"""Checkpoint resume helpers shared by engines_cuda / cpu_train / opencl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

LATEST_NAME = "ckpt_latest.pt"


def resolve_data(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def settings_view(cfg: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    """Subset of cfg used to decide if --resume is allowed."""
    out: Dict[str, Any] = {}
    for k in keys:
        if k not in cfg:
            continue
        v = cfg[k]
        if k == "data":
            out[k] = resolve_data(v)
        elif k == "stages" and isinstance(v, list):
            out[k] = [_stage_key(s) for s in v]
        else:
            out[k] = v
    return out


def _stage_key(s: Any) -> Dict[str, Any]:
    if hasattr(s, "__dict__") and not isinstance(s, dict):
        s = {f: getattr(s, f) for f in ("data_factor", "steps", "max_gaussians_draw", "max_radius", "sh_degree")}
    return {
        "data_factor": int(s["data_factor"]),
        "steps": int(s["steps"]),
        "max_gaussians_draw": int(s["max_gaussians_draw"]),
        "max_radius": int(s["max_radius"]),
        "sh_degree": int(s["sh_degree"]),
    }


def write_cfg(out_dir: Path, cfg: Dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cfg.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def load_cfg(out_dir: Path) -> Dict[str, Any]:
    path = Path(out_dir) / "cfg.json"
    if not path.is_file():
        raise SystemExit(f"--resume needs {path} from a previous run with the same --out")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_same_settings(
    saved: Dict[str, Any],
    current: Dict[str, Any],
    keys: Sequence[str],
) -> None:
    a = settings_view(saved, keys)
    b = settings_view(current, keys)
    mismatches: List[str] = []
    for k in keys:
        if k not in a and k not in b:
            continue
        if a.get(k) != b.get(k):
            mismatches.append(f"  {k}: saved={a.get(k)!r} current={b.get(k)!r}")
    if mismatches:
        raise SystemExit(
            "--resume refused: settings differ from cfg.json in --out.\n"
            "Use the same flags as the interrupted run, or a new --out.\n"
            + "\n".join(mismatches)
        )


def find_checkpoint(out_dir: Path, prefer_latest: bool = True) -> Path:
    out_dir = Path(out_dir)
    latest = out_dir / LATEST_NAME
    if prefer_latest and latest.is_file():
        return latest
    numbered = sorted(out_dir.glob("ckpt_*.pt"))
    numbered = [p for p in numbered if p.name != LATEST_NAME]
    if not numbered:
        # stage-named cpu ckpts
        staged = sorted(out_dir.glob("ckpt_stage*.pt"))
        if staged:
            return staged[-1]
        raise SystemExit(f"--resume: no checkpoint under {out_dir}")
    # Prefer highest step in ckpt_NNNNN.pt
    def step_key(p: Path) -> int:
        stem = p.stem  # ckpt_07000
        parts = stem.split("_")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return -1

    numbered.sort(key=step_key)
    return numbered[-1]


def save_checkpoint(out_dir: Path, payload: Dict[str, Any], also_named: Optional[Path] = None) -> Path:
    """Write ckpt_latest.pt and optionally a numbered/stage file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / LATEST_NAME
    torch.save(payload, latest)
    if also_named is not None:
        also_named = Path(also_named)
        also_named.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, also_named)
        return also_named
    return latest


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def tensor_to_param(t: torch.Tensor, device: torch.device) -> torch.nn.Parameter:
    return torch.nn.Parameter(t.detach().to(device).float())
