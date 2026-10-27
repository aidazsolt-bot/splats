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
from engines_common.resume import load_checkpoint, save_checkpoint


class CpuResumeTests(unittest.TestCase):
    def test_restores_mid_stage_and_continues_next_stage(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            cfg = trainer.TrainConfig(
                data=tmp, out=tmp, resume=True, random_bkgd=False,
                densify_every=0, ckpt_every=1, lpips_lambda=0,
                stages=[trainer.Stage(1, 2, 10, 4, 0), trainer.Stage(1, 1, 10, 4, 0)],
            )
            init = SimpleNamespace(
                means=torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.1, 2.0]]),
                scales=torch.full((2, 3), -2.0),
                quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
                opacities=torch.zeros(2), colors_dc=torch.zeros(2, 3),
                colors_rest=None, scene_scale=1.0,
            )
            model = trainer.GaussianModel(init, cfg.sh_degree_max)
            optimizers = trainer._optimizers(model, cfg)
            sum(p.sum() for p in (
                model.means, model.scales, model.quats, model.opacities,
                model.colors_dc, model.colors_rest,
            )).backward()
            for opt in optimizers.values():
                opt.step()
            save_checkpoint(Path(tmp), trainer._pack_cpu_ckpt(
                cfg, model, stage_i=0, local_step=1, global_step=1,
                optimizers=optimizers,
            ))
            sample = {
                "image": torch.ones(8, 8, 3),
                "alpha": torch.ones(8, 8),
                "K": torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]),
                "viewmat": torch.eye(4), "height": 8, "width": 8,
            }
            with (
                patch.object(trainer, "ColmapRGBADataset", return_value=[sample]),
                patch.object(trainer, "init_gaussians_from_points3d") as initialize,
                patch.object(
                    trainer, "OptionalLPIPS", return_value=SimpleNamespace(available=False)
                ),
            ):
                trainer.train(cfg)
                initialize.assert_not_called()
                saved = load_checkpoint(Path(tmp) / "ckpt_latest.pt")
                self.assertEqual(
                    (saved["stage_i"], saved["local_step"], saved["global_step"]),
                    (1, 1, 3),
                )
                self.assertFalse(torch.equal(saved["means"], model.means))
                means_state = next(iter(saved["optimizers"]["means"]["state"].values()))
                self.assertEqual(means_state["step"].item(), 3)
                trainer.train(cfg)
                cfg.stages[0].steps += 1
                with self.assertRaisesRegex(SystemExit, "stages"):
                    trainer.train(cfg)


if __name__ == "__main__":
    unittest.main()
