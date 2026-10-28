import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import trainer
from engines_common.resume import load_checkpoint


class TinyDataset:
    points = None
    preload_calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 1

    def preload(self, *args, **kwargs):
        TinyDataset.preload_calls += 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return {
            "image": torch.ones(8, 8, 3),
            "alpha": torch.ones(8, 8),
            "K": torch.eye(3),
            "viewmat": torch.eye(4),
            "height": 8,
            "width": 8,
        }


def rasterize(model, viewmat, K, width, height, *args, **kwargs):
    value = torch.sigmoid(sum(p.mean() for p in model.parameters_dict().values()))
    return value.expand(1, height, width, 3), value.expand(1, height, width), {}


def strategy_rasterize(model, viewmat, K, width, height, *args, **kwargs):
    means2d = model.means[:, :2].unsqueeze(0)
    if kwargs.get("absgrad"):
        means2d.absgrad = torch.ones_like(means2d)
    value = torch.sigmoid(means2d.mean())
    colors = value.expand(1, height, width, 3)
    alphas = value.expand(1, height, width)
    info = {
        "means2d": means2d,
        "radii": torch.ones(1, model.means.shape[0], 2),
        "gaussian_ids": torch.arange(model.means.shape[0]),
        "width": width,
        "height": height,
        "n_cameras": 1,
    }
    return colors, alphas, info


def make_init(n):
    return SimpleNamespace(
        means=torch.ones(n, 3), scales=torch.zeros(n, 3),
        quats=torch.ones(n, 4), opacities=torch.full((n,), -2.0),
        colors_dc=torch.zeros(n, 3), colors_rest=None, scene_scale=1.0,
    )


class CudaResumeTests(unittest.TestCase):
    def test_strategy_defaults_and_absgrad_threshold(self):
        default = trainer.TrainConfig(data="data", out="out")
        trainer._normalize_strategy_config(default, None)
        self.assertEqual(default.strategy, "default")
        self.assertEqual(default.refine_stop, 15000)
        self.assertEqual(default.grow_grad2d, 0.0002)

        absgrad = trainer.TrainConfig(data="data", out="out", strategy="default", absgrad=True)
        trainer._normalize_strategy_config(absgrad, None)
        self.assertEqual(absgrad.grow_grad2d, 0.0008)

        mcmc = trainer.TrainConfig(data="data", out="out", strategy="mcmc")
        trainer._normalize_strategy_config(mcmc, None)
        self.assertEqual(mcmc.refine_stop, 25000)
        self.assertEqual(mcmc.cap_max, 200000)
        self.assertEqual(mcmc.opacity_reg, 0.01)

    def test_restores_parameters_optimizer_and_next_step(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            cfg = trainer.TrainConfig(
                data=tmp, out=tmp, max_steps=3, sh_degree=1, densify=False,
                random_bkgd=False, eval_every=0, ckpt_every=1, resume=True,
            )
            init = SimpleNamespace(
                means=torch.ones(2, 3), scales=torch.zeros(2, 3),
                quats=torch.ones(2, 4), opacities=torch.zeros(2),
                colors_dc=torch.zeros(2, 3), colors_rest=None, scene_scale=1.0,
            )
            model = trainer.GaussianModel(init, sh_degree=1)
            optimizers = trainer._build_optimizers(model, cfg, 1.0)
            sum(p.sum() for p in model.parameters_dict().values()).backward()
            for opt in optimizers.values():
                opt.step()
            trainer._save_cuda_ckpt(Path(tmp), cfg, model, 1, optimizers, named=False)
            original_means = model.means.detach().clone()
            with (
                patch.object(trainer, "_require_cuda", return_value=torch.device("cpu")),
                patch.dict(sys.modules, {"gsplat": SimpleNamespace()}),
                patch.object(trainer, "ColmapRGBADataset", TinyDataset),
                patch.object(trainer, "init_gaussians_from_points3d") as initialize,
                patch.object(trainer, "_rasterize", side_effect=rasterize) as render,
                patch("builtins.print") as log,
            ):
                trainer.train(cfg)
                self.assertEqual(TinyDataset.preload_calls, 0)
                messages = [str(call.args[0]) for call in log.call_args_list if call.args]
                self.assertTrue(any("Loading checkpoint" in message for message in messages))
                self.assertTrue(any("First gsplat rasterization" in message for message in messages))
                for step in (2, 3):
                    progress = [
                        call for call in log.call_args_list
                        if call.args and str(call.args[0]).startswith(f"step={step}/3 ")
                    ]
                    self.assertEqual(len(progress), 1)
                    self.assertTrue(progress[0].kwargs["flush"])
                initialize.assert_not_called()
                self.assertEqual(render.call_count, 2)
                saved = load_checkpoint(Path(tmp) / "ckpt_latest.pt")
                self.assertEqual(saved["step"], 3)
                self.assertFalse(torch.equal(saved["means"], original_means))
                for optimizer in saved["optimizers"].values():
                    state = next(iter(optimizer["state"].values()))
                    self.assertEqual(state["step"].item(), 3)
                render.reset_mock()
                trainer.train(cfg)
                render.assert_not_called()
                cfg.cache_images = True
                trainer.train(cfg)
                self.assertEqual(TinyDataset.preload_calls, 2)
                cfg.data_factor += 1
                with self.assertRaisesRegex(SystemExit, "data_factor"):
                    trainer.train(cfg)
                self.assertEqual(load_checkpoint(Path(tmp) / "ckpt_latest.pt")["step"], 3)

    def test_default_strategy_refines_and_persists_state(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            cfg = trainer.TrainConfig(
                data=tmp, out=tmp, max_steps=2, save_steps=(2,), sh_degree=1,
                densify=True, random_bkgd=False, eval_every=0, ckpt_every=0,
                strategy="default", refine_start=0, refine_stop=2, refine_every=1,
                grow_grad2d=1e-12, reset_every=100, absgrad=True, antialiased=True,
            )
            with (
                patch.object(trainer, "_require_cuda", return_value=torch.device("cpu")),
                patch.object(trainer, "ColmapRGBADataset", TinyDataset),
                patch.object(trainer, "init_gaussians_from_points3d", return_value=make_init(2)),
                patch.object(trainer, "_rasterize", side_effect=strategy_rasterize) as render,
                patch("builtins.print"),
            ):
                trainer.train(cfg)
            self.assertTrue(render.call_args.kwargs["absgrad"])
            self.assertEqual(render.call_args.kwargs["rasterize_mode"], "antialiased")
            saved = load_checkpoint(Path(tmp) / "ckpt_latest.pt")
            self.assertEqual(saved["cfg"]["strategy"], "default")
            self.assertGreater(saved["means"].shape[0], 2)
            self.assertIsNotNone(saved["strategy_state"]["grad2d"])
            self.assertEqual(saved["strategy_state"]["count"].device.type, "cpu")

    def test_mcmc_strategy_respects_cap_and_persists_state(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            cfg = trainer.TrainConfig(
                data=tmp, out=tmp, max_steps=2, save_steps=(2,), sh_degree=1,
                densify=True, random_bkgd=False, eval_every=0, ckpt_every=0,
                strategy="mcmc", refine_start=0, refine_stop=2, refine_every=1,
                cap_max=22,
            )
            with (
                patch.object(trainer, "_require_cuda", return_value=torch.device("cpu")),
                patch.object(trainer, "ColmapRGBADataset", TinyDataset),
                patch.object(trainer, "init_gaussians_from_points3d", return_value=make_init(20)),
                patch.object(trainer, "_rasterize", side_effect=strategy_rasterize),
                patch(
                    "gsplat.strategy.ops.compute_relocation",
                    side_effect=lambda opacities, scales, ratios, binoms: (
                        opacities.clamp(max=0.99),
                        scales / ratios.to(scales.dtype).unsqueeze(-1).sqrt(),
                    ),
                ),
                patch("gsplat.strategy.mcmc.inject_noise_to_position"),
                patch("builtins.print"),
            ):
                trainer.train(cfg)
            saved = load_checkpoint(Path(tmp) / "ckpt_latest.pt")
            self.assertEqual(saved["cfg"]["strategy"], "mcmc")
            self.assertEqual(saved["cfg"]["opacity_reg"], 0.01)
            self.assertGreater(saved["means"].shape[0], 20)
            self.assertLessEqual(saved["means"].shape[0], 22)
            self.assertEqual(saved["strategy_state"]["binoms"].device.type, "cpu")

    def test_default_strategy_state_is_restored_on_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            cfg = trainer.TrainConfig(
                data=tmp, out=tmp, max_steps=2, save_steps=(2,), sh_degree=1,
                densify=True, random_bkgd=False, eval_every=0, ckpt_every=0,
                strategy="default", refine_start=0, refine_stop=3, refine_every=100,
                grow_grad2d=1e-12, reset_every=100,
            )
            trainer._normalize_strategy_config(cfg, None)
            init = make_init(2)
            model = trainer.GaussianModel(init, sh_degree=1)
            optimizers = trainer._build_optimizers(model, cfg, model.scene_scale)
            from gsplat import DefaultStrategy

            strategy = DefaultStrategy(
                refine_start_iter=cfg.refine_start,
                refine_stop_iter=cfg.refine_stop,
                refine_every=cfg.refine_every,
            )
            state = strategy.initialize_state(model.scene_scale)
            state["grad2d"] = torch.tensor([2.0, 3.0])
            state["count"] = torch.tensor([4.0, 5.0])
            trainer._save_cuda_ckpt(
                Path(tmp), cfg, model, 1, optimizers, named=False,
                strategy_state=state,
            )
            cfg.resume = True
            with (
                patch.object(trainer, "_require_cuda", return_value=torch.device("cpu")),
                patch.object(trainer, "ColmapRGBADataset", TinyDataset),
                patch.object(trainer, "init_gaussians_from_points3d") as initialize,
                patch.object(trainer, "_rasterize", side_effect=strategy_rasterize),
                patch("builtins.print"),
            ):
                trainer.train(cfg)
            initialize.assert_not_called()
            saved = load_checkpoint(Path(tmp) / "ckpt_latest.pt")
            self.assertEqual(saved["step"], 2)
            self.assertTrue(torch.equal(saved["strategy_state"]["count"], torch.tensor([5.0, 6.0])))


if __name__ == "__main__":
    unittest.main()
