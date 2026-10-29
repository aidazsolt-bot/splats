"""Colab setup and restart-safe execution for the pinned CUDA comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT = Path("/content")
REPO = ROOT / "splats"
VENV = ROOT / "splats-cu134-env"
PYTHON = VENV / "bin" / "python"
PYTHON_VERSION = "3.12.9"
TORCH_VERSION = "2.15.0+cu134"
NVCC_VERSION = "13.4.92"
TOOLKIT_VERSION = "13.4.3"
GSPLAT_COMMIT = "2a29ad06ba008bb34c9b942e82c2d46d39a194f4"
NINJA_VERSION = "1.13.2"
MAX_JOBS = 10


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
        BUILD_NO_CUDA="1",
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
        ROOT / "torch_extensions" / f"cu134-{GSPLAT_COMMIT}-sm{actual['arch'].replace('.', '')}"
    )
    print(
        f"Python {actual['python']} | torch {actual['torch']} | nvcc {NVCC_VERSION} | "
        f"gsplat {actual['gsplat']} ({GSPLAT_COMMIT[:8]})\n"
        f"GPU: {actual['gpu']} | arch {actual['arch']} | MAX_JOBS={MAX_JOBS}\n"
        f"Extension cache: {env['TORCH_EXTENSIONS_DIR']}",
        flush=True,
    )
    return Runtime(env)


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
        "install", f"cuda-toolkit[nvcc,cudart,cccl]=={TOOLKIT_VERSION}",
        f"ninja=={NINJA_VERSION}", "setuptools", "wheel",
    ])
    constraints = VENV / "comparison-constraints.txt"
    constraints.write_text(f"torch=={TORCH_VERSION}\nninja=={NINJA_VERSION}\n", encoding="utf-8")
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
