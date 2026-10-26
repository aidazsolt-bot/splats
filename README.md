# splats

Macro Gaussian splat tooling for COLMAP + RGBA datasets (e.g. Blackberry).

## Pipelines

- [`engines/`](engines/) - CPU, deterministic, no training
- [`engines_cuda/`](engines_cuda/) - PyTorch + gsplat training (needs NVIDIA CUDA)
- [`engines_cpu_train/`](engines_cpu_train/) - pure PyTorch training on CPU (quality-first schedule)

## Usage

### CPU quality training (no CUDA)

```bash
pip install -r engines_cpu_train/requirements.txt
python engines_cpu_train/run_train.py --data "$HOME/tmp/blackberry-splats/dataset" --out "$HOME/tmp/blackberry-splats/engines_cpu_train_out"
```

Smoke test: `python engines_cpu_train/run_train.py --smoke`

Details: [`engines_cpu_train/README.md`](engines_cpu_train/README.md)

### CUDA / Torch training (GPU machine)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r engines_cuda/requirements.txt
python engines_cuda/run_train.py --data "$HOME/tmp/blackberry-splats/dataset" --max-steps 30000
```

Match the Torch wheel CUDA tag (`cu121`, `cu124`, ...) to the driver/toolkit on that machine.

Optional flags: `--data-factor 2`, `--out "$HOME/tmp/blackberry-splats/engines_cuda_out"`, `--smoke` (short verification run).

Details: [`engines_cuda/README.md`](engines_cuda/README.md)

### CPU deterministic (no training)

```bash
pip install -r engines/requirements.txt
python engines/run_blackberry.py --data "$HOME/tmp/blackberry-splats/dataset" --out "$HOME/tmp/blackberry-splats/engines_out"
```

Details: [`engines/README.md`](engines/README.md)
