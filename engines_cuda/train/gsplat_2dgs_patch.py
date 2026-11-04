"""Work around gsplat 2DGS projection backward contiguous-grad bug.

Upstream: nerfstudio-project/gsplat#1098
(projection_2dgs_*_bwd CHECK_INPUT requires contiguous grads; autograd can
hand in non-contiguous v_depths when normal/distortion losses are used).

Patch _make_lazy_cuda_func so fused/packed 2DGS bwd ops contiguous their
gradient tensors before the CUDA check. Safe to call more than once.
"""

from __future__ import annotations

_PATCHED = False


def install_2dgs_contiguous_grad_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import torch
    from gsplat.cuda import _wrapper as gw

    orig = gw._make_lazy_cuda_func
    bwd_ops = frozenset(
        {
            "projection_2dgs_fused_bwd",
            "projection_2dgs_packed_bwd",
        }
    )

    def _patched_make(name: str):
        fn = orig(name)
        if name not in bwd_ops:
            return fn

        def call_cuda(*args, **kwargs):
            args = list(args)
            # Gradients sit just before the trailing viewmats_requires_grad bool.
            # Contiguous every tensor in that tail so CHECK_INPUT passes.
            end = len(args)
            if end > 0 and isinstance(args[-1], bool):
                end -= 1
            for i in range(end - 1, -1, -1):
                t = args[i]
                if not isinstance(t, torch.Tensor):
                    break
                if not t.is_contiguous():
                    args[i] = t.contiguous()
            return fn(*args, **kwargs)

        return call_cuda

    gw._make_lazy_cuda_func = _patched_make
    _PATCHED = True
