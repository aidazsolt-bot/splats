# Windows: cl.exe and gsplat CUDA builds

This setup applies to branches containing `engines_cuda`. CPU-only and
non-training pipelines do not require MSVC or CUDA.

`cl.exe` is the Microsoft C++ compiler, not part of the CUDA toolkit.
`where.exe cl` failing means MSVC is either missing or its developer environment
has not been activated. Installing Ninja alone does not fix that.

## Install MSVC if missing

Install Visual Studio 2022 Build Tools with Desktop development with C++,
MSVC x64 tools, and the Windows SDK. In PowerShell:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-package-agreements --accept-source-agreements --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Wait for the installation to finish and accept elevation if requested.
For an existing installation, use Visual Studio Installer > Modify to add
the C++ workload if it is missing.

## Activate MSVC in the training terminal

A new ordinary PowerShell does **not** automatically activate MSVC. Either use
Developer PowerShell for VS 2022, or run this in the current PowerShell:

```powershell
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (!(Test-Path $vswhere)) { throw "Install Visual Studio 2022 Build Tools first." }
$vs = & $vswhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw "MSVC x64 tools missing; modify Build Tools in Visual Studio Installer." }
& "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
where.exe cl
cl
```

`cl` without source files prints compiler information and a missing-source
diagnostic; that is expected. Do not copy just `cl.exe` onto PATH: the developer
environment also configures the headers, libraries, and Windows SDK.

## Check Ninja and CUDA, then train in the same terminal

Use the same Python interpreter for installing dependencies and training:

```powershell
python.exe -m pip install ninja
$ninjaBin = python.exe -c "import ninja; print(ninja.BIN_DIR)"
$env:Path = "$ninjaBin;$env:Path"
ninja --version
nvcc --version
python.exe -c "import torch; print(torch.__version__, torch.version.cuda)"
python.exe -c "from gsplat.cuda._backend import _C; assert _C is not None"
```

The CUDA toolkit must match the CUDA version used by PyTorch, and the MSVC
version must be supported by that toolkit. The gsplat check may compile CUDA
kernels on first use. These shell environment changes last only for this
terminal.

From the repository root on a branch containing `engines_cuda`:

```powershell
python.exe .\engines_cuda\run_train.py --resume --data X:\blackberry-splats\dataset --out cuda_out --data-factor 4
```

The trainer immediately prints startup phases and the selected NVIDIA device.
The first gsplat rasterization may compile its CUDA extension on the CPU; the
message before that call does not confirm that compilation is actually running.
Step metrics are flushed after the first completed step (also after resume),
every 100 steps, at the final step, or after 10 seconds at the next completed
step. A single slow step or extension build can still take longer without a
new metric line.

To check actual GPU activity independently, run `nvidia-smi -l 1` in another
terminal. CUDA device numbering is separate from Task Manager numbering:
`cuda:0` can be Task Manager's GPU 1. Intel UHD is not a CUDA device.
