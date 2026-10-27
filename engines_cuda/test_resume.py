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


def rasterize(model, viewmat, K, width, height, *args):
    value = torch.sigmoid(sum(p.mean() for p in model.parameters_dict().values()))
    return value.expand(1, height, width, 3), value.expand(1, height, width), {}


class CudaResumeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
