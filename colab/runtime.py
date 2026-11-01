"""Colab setup and restart-safe execution for the pinned CUDA comparison."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

ROOT = Path("/content")
REPO = ROOT / "splats"
VENV = ROOT / "splats-cu134-env"
PYTHON = VENV / "bin" / "python"
PYTHON_VERSION = "3.12.15"
TORCH_VERSION = "2.15.0+cu134"
NVCC_VERSION = "13.4.92"
TOOLKIT_VERSION = "13.4.1"
CCCL_VERSION = "13.3.4.3.1"
GSPLAT_COMMIT = "512d366b67073d77ca099ede742683c165dfc23b"
NINJA_VERSION = "1.13.2"
MAX_JOBS = 10

# Prebuilt full JIT (sm_75 / T4), uploaded from the Mate Linux host.
# Direct-load only: do not drop into ninja and expect a no-op rebuild.
PREBUILT_DRIVE_DIR = (
    ROOT / "drive/MyDrive/blackberry-splats/gsplat-prebuilt"
    / "py312-torch215-cu134-512d366-sm75"
)
PREBUILT_SO_NAME = "gsplat_cuda.so"
PREBUILT_ARCH = "7.5"
PREBUILT_SHA256 = "2efd223259d15bc627acbf16c2f7cfc96a8d42a96aafb516fbf198a080b7b9fc"


def run_live(args: list[str], *, env: dict[str, str] | None = None) -> None:
    """Stream child output into the notebook, rather than Colab server logs."""
    print("$ " + shlex.join(args), flush=True)
    with subprocess.Popen(
        args, cwd=REPO, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        bufsize=1, start_new_session=True,
    ) as process:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
            code = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code != 0:
        if code in (-9, 137):
            print("Process was killed; check Colab host RAM (10 compiler jobs can exceed 12 GB).", flush=True)
        raise subprocess.CalledProcessError(code, args)


def capture(args: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        args, cwd=REPO, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", flush=True)
        raise subprocess.CalledProcessError(result.returncode, args)
    return result.stdout.strip()


def build_environment(cuda: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        CUDA_HOME=str(cuda),
        CUDA_PATH=str(cuda),
        PATH=os.pathsep.join((str(VENV / "bin"), str(cuda / "bin"), env.get("PATH", ""))),
        LD_LIBRARY_PATH=os.pathsep.join((str(cuda / "lib"), env.get("LD_LIBRARY_PATH", ""))),
        CPATH=os.pathsep.join(filter(None, (
            str(cuda / "include" / "cccl"), env.get("CPATH", ""),
        ))),
        BUILD_NO_CUDA="1",
        BUILD_2DGS="0",
        BUILD_3DGUT="0",
        NUM_CHANNELS="3",
        MAX_JOBS=str(MAX_JOBS),
        VERBOSE="1",
        PYTHONUNBUFFERED="1",
        NINJA_STATUS="[%f/%t %es] ",
    )
    return env


class Runtime:
    def __init__(self, env: dict[str, str]):
        self.env = env

    def run(self, args: list[str]) -> None:
        run_live([str(PYTHON), "-u", *args], env=self.env)

    def code(self, source: str) -> None:
        self.run(["-c", source])

    def pip(self, args: list[str]) -> None:
        self.run(["-m", "pip", *args])


def load_runtime() -> Runtime:
    """Reconstruct the child environment without relying on notebook globals."""
    if not PYTHON.is_file():
        raise RuntimeError(
            "The comparison Python environment is missing or its interpreter link is broken. "
            "Run sections 1 and 2 in THIS Colab session, then 2a-2c. "
            "A replacement Colab VM loses /content, including installed packages and build caches; "
            "do not substitute the notebook kernel's Python."
        )
    cuda = Path(capture([
        str(PYTHON), "-c",
        "import importlib.metadata as m; "
        "print(m.distribution('nvidia-cuda-nvcc').locate_file('nvidia/cu13'))",
    ]))
    for relative in (
        "bin/nvcc", "include/cuda_runtime.h", "include/cccl/cub/cub.cuh",
        "nvvm/libdevice/libdevice.10.bc", "lib/libcudart.so.13",
    ):
        if not (cuda / relative).is_file():
            raise RuntimeError(f"Incomplete CUDA toolkit: missing {cuda / relative}. Rerun section 2.")
    if not (cuda / "lib/libcudart.so").exists():
        (cuda / "lib/libcudart.so").symlink_to("libcudart.so.13")
    env = build_environment(cuda)
    nvcc = capture([str(cuda / "bin/nvcc"), "--version"], env=env)
    if f"V{NVCC_VERSION}" not in nvcc:
        raise RuntimeError(f"Expected nvcc {NVCC_VERSION}; got:\n{nvcc}\nRerun section 2.")
    probe = """
import importlib.metadata as m
import json
import platform
import torch

source = json.loads(m.distribution('gsplat').read_text('direct_url.json'))
assert torch.cuda.is_available(), 'CUDA unavailable: check the Colab GPU runtime and driver'
print(json.dumps({
    'python': platform.python_version(),
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'gsplat': m.version('gsplat'),
    'commit': source['vcs_info']['commit_id'],
    'ninja': m.version('ninja'),
    'arch': '.'.join(map(str, torch.cuda.get_device_capability(0))),
    'gpu': torch.cuda.get_device_name(0),
}))
"""
    actual = json.loads(capture([str(PYTHON), "-c", probe], env=env))
    expected = dict(
        python=PYTHON_VERSION, torch=TORCH_VERSION, cuda="13.4",
        gsplat="1.6.0", commit=GSPLAT_COMMIT, ninja=NINJA_VERSION,
    )
    for name, value in expected.items():
        if actual[name] != value:
            raise RuntimeError(f"Expected {name}={value}, got {actual[name]}. Rerun section 2.")
    env["TORCH_CUDA_ARCH_LIST"] = actual["arch"]
    env["TORCH_EXTENSIONS_DIR"] = str(
        ROOT / "torch_extensions" / f"cu134-{GSPLAT_COMMIT}-rgb3dgs-sm{actual['arch'].replace('.', '')}"
    )
    env.pop("GSPLAT_USE_PREBUILT", None)
    print(
        f"Python {actual['python']} | torch {actual['torch']} | nvcc {NVCC_VERSION} | "
        f"gsplat {actual['gsplat']} ({GSPLAT_COMMIT[:8]})\n"
        f"GPU: {actual['gpu']} | arch {actual['arch']} | MAX_JOBS={MAX_JOBS}\n"
        f"Extension cache: {env['TORCH_EXTENSIONS_DIR']}",
        flush=True,
    )
    return Runtime(env)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_prebuilt_extension(runtime: Runtime) -> bool:
    """Copy the Drive prebuilt gsplat_cuda.so into the extension cache.

    Returns True when the file is ready for direct load (skips ninja compile).
    Only for compute capability 7.5 (T4 / GTX 1650 class). Returns False so the
    caller can fall back to a local restricted JIT build.
    """
    arch = runtime.env.get("TORCH_CUDA_ARCH_LIST", "")
    if arch != PREBUILT_ARCH:
        print(
            f"Prebuilt gsplat_cuda.so is sm_75 only (got arch {arch}); "
            "falling back to local compile.",
            flush=True,
        )
        return False

    drive_so = PREBUILT_DRIVE_DIR / PREBUILT_SO_NAME
    if not drive_so.is_file():
        print(
            f"Prebuilt missing at {drive_so}. Mount Drive and ensure the file "
            "is under blackberry-splats/gsplat-prebuilt/; falling back to compile.",
            flush=True,
        )
        return False

    meta_path = PREBUILT_DRIVE_DIR / "META.json"
    expected = PREBUILT_SHA256
    if meta_path.is_file():
        try:
            expected = json.loads(meta_path.read_text(encoding="utf-8")).get(
                "sha256", expected
            )
        except json.JSONDecodeError:
            pass

    digest = _sha256_file(drive_so)
    if digest != expected:
        raise RuntimeError(
            f"Prebuilt SHA-256 mismatch for {drive_so}: got {digest}, expected {expected}"
        )

    cache = ROOT / "torch_extensions" / f"cu134-{GSPLAT_COMMIT}-prebuilt-sm75"
    dest_dir = cache / "gsplat_cuda"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / PREBUILT_SO_NAME
    if not dest.is_file() or dest.stat().st_size != drive_so.stat().st_size:
        print(f"Copying prebuilt {drive_so} -> {dest} ...", flush=True)
        shutil.copy2(drive_so, dest)
    elif _sha256_file(dest) != expected:
        print(f"Refreshing prebuilt cache {dest} ...", flush=True)
        shutil.copy2(drive_so, dest)

    if _sha256_file(dest) != expected:
        raise RuntimeError(f"Cached prebuilt SHA-256 mismatch: {dest}")

    runtime.env["TORCH_EXTENSIONS_DIR"] = str(cache)
    runtime.env["GSPLAT_USE_PREBUILT"] = "1"
    # Avoid accidental restricted JIT if something still calls build_and_load.
    for key in ("BUILD_2DGS", "BUILD_3DGUT", "NUM_CHANNELS"):
        runtime.env.pop(key, None)
    print(
        f"PASS: prebuilt gsplat_cuda.so ready ({dest.stat().st_size} bytes, sm_75)\n"
        f"Extension cache: {cache}",
        flush=True,
    )
    return True


# Snippet for the training venv: inject prebuilt as gsplat.csrc BEFORE import gsplat.
# Top-level `import gsplat` pulls cuda._wrapper -> _backend and would JIT otherwise.
LOAD_PREBUILT_SNIPPET = r"""
import os
import sys
import time
import resource
from pathlib import Path

import torch  # noqa: F401
from torch.utils.cpp_extension import _import_module_from_library

started = time.perf_counter()
print('Starting gsplat load...', flush=True)
try:
    use_prebuilt = os.environ.get('GSPLAT_USE_PREBUILT') == '1'
    ext_dir = Path(os.environ['TORCH_EXTENSIONS_DIR']) / 'gsplat_cuda'
    so_path = ext_dir / 'gsplat_cuda.so'
    if not (use_prebuilt and so_path.is_file()):
        raise RuntimeError(
            f'Prebuilt load expected but missing (GSPLAT_USE_PREBUILT={use_prebuilt!r}, so={so_path})'
        )
    print(f'Loading prebuilt {so_path} ...', flush=True)
    mod = _import_module_from_library('gsplat_cuda', str(ext_dir), True)
    # Must be registered before any `import gsplat` (which imports cuda._backend).
    sys.modules['gsplat.csrc'] = mod
    import gsplat
    gsplat.csrc = mod
    from gsplat.cuda import _backend as _backend_mod
    from gsplat.cuda._backend import _C
    if _C is None or getattr(_backend_mod, '_C', None) is None:
        raise RuntimeError('gsplat CUDA extension unavailable after prebuilt inject')
    # If JIT still ran, _backend would have replaced _C via build_and_load; refuse that path.
    if getattr(_C, '__file__', None) and 'torch_extensions' in str(getattr(_C, '__file__', '')):
        pass  # loaded from our extension cache
    print('PASS: gsplat build/load', flush=True)
    print('Mode: prebuilt Drive .so (no ninja compile)', flush=True)
    print(f'Extension module: {getattr(_C, "__file__", type(_C))}', flush=True)
finally:
    print(f'Build/load elapsed: {time.perf_counter() - started:.1f} s', flush=True)
    print(f'Python peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MiB', flush=True)
    print(f'Largest reaped child peak RSS: {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024:.1f} MiB (not combined process-tree memory)', flush=True)
"""


def install() -> None:
    if sys.platform != "linux":
        raise RuntimeError("This setup is for a Linux Colab GPU runtime.")
    bootstrap = ROOT / "splats-bootstrap"
    run_live([
        sys.executable, "-m", "pip", "install", "--upgrade", "--target", str(bootstrap), "uv==0.13.6",
    ])
    uv = str(bootstrap / "bin" / "uv")
    run_live([uv, "python", "install", PYTHON_VERSION])
    base = capture([uv, "python", "find", "--managed-python", PYTHON_VERSION])
    version = capture([base, "-c", "import platform; print(platform.python_version())"])
    if version != PYTHON_VERSION:
        raise RuntimeError(f"Expected Python {PYTHON_VERSION}, got {version}.")
    run_live([base, "-m", "venv", str(VENV)])
    env = os.environ.copy()
    env.update(BUILD_NO_CUDA="1", MAX_JOBS=str(MAX_JOBS), PYTHONUNBUFFERED="1")
    runtime = Runtime(env)
    runtime.pip(["install", f"torch=={TORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/cu134"])
    runtime.pip([
        "install", f"cuda-toolkit[cudart]=={TOOLKIT_VERSION}",
        f"nvidia-cuda-nvcc=={NVCC_VERSION}", f"nvidia-cuda-crt=={NVCC_VERSION}",
        f"nvidia-nvvm=={NVCC_VERSION}", f"nvidia-cuda-cccl=={CCCL_VERSION}",
        f"ninja=={NINJA_VERSION}", "setuptools", "wheel",
    ])
    constraints = VENV / "comparison-constraints.txt"
    constraints.write_text(
        f"torch=={TORCH_VERSION}\nninja=={NINJA_VERSION}\n"
        f"cuda-toolkit=={TOOLKIT_VERSION}\nnvidia-cuda-nvcc=={NVCC_VERSION}\n"
        f"nvidia-cuda-crt=={NVCC_VERSION}\nnvidia-nvvm=={NVCC_VERSION}\n"
        f"nvidia-cuda-cccl=={CCCL_VERSION}\n",
        encoding="utf-8",
    )
    runtime.pip(["install", "-c", str(constraints), "-r", "engines_cuda/requirements.txt"])
    runtime.pip([
        "install", "-c", str(constraints), "--no-build-isolation", "--force-reinstall", "--no-deps",
        f"gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@{GSPLAT_COMMIT}",
    ])
    runtime.pip(["install", "-c", str(constraints), "gsplat==1.6.0"])
    runtime.pip(["check"])
    load_runtime()
    print("PASS: installation and version checks. Continue with 2a.", flush=True)


def save_paths(data: str, out: str) -> None:
    (ROOT / "splats-dataset-paths.json").write_text(
        json.dumps({"data": data, "out": out}), encoding="utf-8",
    )


def load_paths() -> tuple[str, str]:
    config = ROOT / "splats-dataset-paths.json"
    if not config.is_file():
        raise RuntimeError("Dataset paths are not configured in this VM. Run section 3 to mount Drive.")
    paths = json.loads(config.read_text(encoding="utf-8"))
    data, out = paths["data"], paths["out"]
    for relative in ("cameras.txt", "images.txt", "points3D.txt", "images"):
        if not (Path(data) / relative).exists():
            raise RuntimeError(f"Missing dataset entry {Path(data) / relative}. Run section 3 to mount/check Drive.")
    Path(out).mkdir(parents=True, exist_ok=True)
    return data, out
