# Gaussian densification strategy plan

**Status:** Implemented after the factor-2 run in `cuda_out_2` was stopped.

## Decisions

- New runs should default to `--strategy default`.
- MCMC should default to `--cap-max 200000`, matching the Gaussian count in
  `danny_splat_45000.ply`.
- Existing output folders whose `cfg.json` has no strategy key should resume
  with `legacy`.

## Why replace the current strategy?

The current `_simple_densify_and_prune` in `train/trainer.py`:

- Uses the 3D means gradient averaged over all steps, with threshold `0.0002`
  intended for a 2D screen-space gradient. It barely densifies: the run had
  38,288 Gaussians at step 1 and 37,166 at step 20,000.
- Does not prune oversized Gaussians, which may contribute to streaks and
  floaters.
- Recreates Adam after densification, losing optimizer momentum for existing
  Gaussians.
- Rebuilds the means learning-rate scheduler after densification, causing the
  learning rate to decay more than intended.
- Resets opacity every 3,000 steps while pruning low-opacity Gaussians. This
  could contribute to the user's observation that the berries became
  transparent at step 15,000; this cause has not been verified.

## CLI design

### Strategy selection and shared options

- `--strategy {legacy,default,mcmc}`; default `default` for new runs.
- `--refine-start 500`
- `--refine-stop 15000` for `default`; `25000` for `mcmc`
- `--refine-every 100`
- `--prune-opa 0.005`
- `--no-densify` disables refinement for all strategies.
- `--antialiased` enables `rasterize_mode="antialiased"` (Mip-Splatting-style
  rendering) for all strategies.

### `default`

- `--grow-grad2d 0.0002`; use `0.0008` with `--absgrad`.
- `--absgrad`: use absolute gradients for refinement.
- `--grow-scale3d 0.01`
- `--prune-scale3d 0.1`
- `--reset-every 3000`
- `--pause-refine-after-reset 0`
- `--revised-opacity`: use the experimental opacity adjustment when splitting.

### `mcmc`

- `--cap-max 200000`
- `--noise-lr 5e5`
- `--opacity-reg 0.01`
- `--scale-reg 0.01`

MCMC receives the current means learning rate for its noise schedule.

## Implementation and compatibility

- Add strategy selection and strategy parameters to `CUDA_RESUME_KEYS`, so
  resume rejects incompatible strategy settings.
- Treat old `cfg.json` files without a strategy key as `legacy`.
- Save and restore strategy state (such as accumulated gradients and
  visibility counts) in checkpoints.
- Use the parameter dictionary and optimizer mapping expected by gsplat.
  Call `step_pre_backward` and `step_post_backward`; pass `absgrad` and
  `rasterize_mode` to rasterization where appropriate.
- Do not rebuild the means learning-rate scheduler after refinement for
  `default` or `mcmc`; preserve existing legacy behavior.
- Add mocked-rasterizer tests for each strategy and a resume test covering
  strategy state.
- Document recommendations for the GTX 1650 in the README.

Implemented in `train/trainer.py` and `run_train.py`; validated with mocked
rasterizer tests for legacy resume, default densification, MCMC cap handling,
and default-strategy state restore. GTX 1650 MCMC cap remains hardware/data
dependent and requires a real run to benchmark.

## Reference: Dany Bittel's workflow and model

Source: [Dany Bittel, Macro Splats 2025](https://danybittel.ch/macro.html).
The dataset includes a CC-BY license in `extract/LICENSE.txt`.

- Focus stacks 16 shots per angle at f/18 with Zerene Stacker; estimates camera
  poses using COLMAP; applies color correction and background masking; trains
  with Postshot (Jawset); manually retouches the mounting.
- `danny_splat_45000.ply` contains exactly 200,000 Gaussians. This count is
  consistent with a fixed cap, but it is **not confirmed** that this was a
  Postshot setting. The filename suggests 45,000 steps, also unconfirmed.
- The PLY has median opacity 1.0, degree-3 SH coefficients in use, and all-zero
  normals.
- First reproduction candidate: `--strategy mcmc --cap-max 200000
  --max-steps 45000`.

## Other strategies and methods

These alternatives were not independently verified online when this plan was
written; treat the availability and summaries below as tentative.

| Method | Idea | Availability |
|---|---|---|
| AbsGS | Absolute gradients to improve fine-detail densification | gsplat exposes `absgrad` |
| Revising Densification | Opacity correction when splitting Gaussians | gsplat exposes `revised_opacity` |
| 3DGS-MCMC | Fixed budget; relocates Gaussians rather than simply deleting them | gsplat `MCMCStrategy` |
| Mip-Splatting | Anti-aliasing across zoom and resolution changes | gsplat `rasterize_mode="antialiased"` |
| Taming 3DGS | Fixed budget with score-based densification | Separate implementation; consider later |
| Mini-Splatting | Fewer Gaussians at similar quality | Separate implementation; consider later |
| Improved-GS | Alternative densification improvements | Separate implementation; consider later |

## Alternative training tools

| Tool | Type | Notes |
|---|---|---|
| Postshot (Jawset) | Commercial GUI; NVIDIA GPU | Used in Dany Bittel's published workflow |
| LichtFeld Studio | Open-source C++/CUDA GUI | Fast local trainer; includes MCMC and a Gaussian cap |
| Brush | Open-source Rust/WebGPU | Avoids CUDA/compiler setup; supports Intel/AMD GPUs |
| OpenSplat | Open-source C++ | Supports CPU, CUDA, and Metal |

All are reported to accept COLMAP data. Alpha-mask behavior varies by tool and
has not been verified here.
