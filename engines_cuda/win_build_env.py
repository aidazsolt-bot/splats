"""Prepare a Windows environment for gsplat's JIT CUDA extension build.

gsplat ships no prebuilt Windows wheels for Python 3.12, so it compiles its CUDA
extension on first use. On Windows that needs MSVC (``cl.exe``) on PATH, and
gsplat passes the GCC-only flag ``-Wno-attributes`` which MSVC rejects (D8021).
Call :func:`setup` before the first gsplat rasterization.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_DROP_CFLAGS = {"-Wno-attributes"}
# MSVC STL >= 17.10 refuses CUDA < 12.4 (STL1002); nvcc < 12.4 refuses MSVC >= 17.10.
_COMPAT_CFLAGS = [
    "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
    "/Zc:preprocessor",
]
_COMPAT_CUDA_CFLAGS = [
    "-allow-unsupported-compiler",
    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
    "-Xcompiler=/Zc:preprocessor",
]


def _find_vcvars() -> Path | None:
    vswhere = (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if not vswhere.is_file():
        return None
    try:
        out = subprocess.check_output(
            [
                str(vswhere), "-latest", "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        bat = Path(line.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if bat.is_file():
            return bat
    return None


def _activate_msvc() -> None:
    if shutil.which("cl"):
        return
    bat = _find_vcvars()
    if bat is None:
        raise RuntimeError(
            "MSVC (cl.exe) not found. Install Visual Studio 2022 Build Tools with "
            "'Desktop development with C++' (see WINDOWS_CUDA_SETUP.md)."
        )
    out = subprocess.check_output(
        f'cmd /s /c ""{bat}" >nul && set"', text=True, errors="replace"
    )
    for line in out.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            os.environ[key] = value
    if not shutil.which("cl"):
        raise RuntimeError(f"cl.exe still not on PATH after running {bat}")
    print(f"[win_build_env] activated MSVC from {bat}")


def _patch_jit_flags() -> None:
    import torch.utils.cpp_extension as ce

    if getattr(ce._jit_compile, "_splats_patched", False):
        return
    orig = ce._jit_compile

    def _jit_compile(name, sources, extra_cflags, extra_cuda_cflags, *args, **kwargs):
        cflags = [f for f in (extra_cflags or []) if f not in _DROP_CFLAGS]
        cflags += [f for f in _COMPAT_CFLAGS if f not in cflags]
        cuda_cflags = list(extra_cuda_cflags or [])
        cuda_cflags += [f for f in _COMPAT_CUDA_CFLAGS if f not in cuda_cflags]
        return orig(name, sources, cflags, cuda_cflags, *args, **kwargs)

    _jit_compile._splats_patched = True
    ce._jit_compile = _jit_compile


def setup() -> None:
    """No-op outside Windows; must run before ``gsplat.cuda._backend`` is imported."""
    if sys.platform != "win32":
        return
    if "gsplat.cuda._backend" in sys.modules:
        return
    os.environ.setdefault("MAX_JOBS", "1")
    _activate_msvc()
    _patch_jit_flags()
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
