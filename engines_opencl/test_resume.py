import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from engines_opencl import run_train as trainer
from engines_common.resume import load_checkpoint, save_checkpoint


class OpenCLResumeTests(unittest.TestCase):
    def test_cpu_fallback_restores_optimizer_and_step(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            cfg = {
                "data": tmp, "out": tmp, "steps": 3, "data_factor": 4,
                "seed": 42, "prefer": "nvidia", "cpu_fallback": True,
                "smoke": False, "device_index": None,
            }
            parameters = [
                torch.nn.Parameter(torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.1, 2.0]])),
                torch.nn.Parameter(torch.full((2, 3), -2.0)),
                torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)),
                torch.nn.Parameter(torch.zeros(2)),
                torch.nn.Parameter(torch.zeros(2, 3)),
                torch.nn.Parameter(torch.zeros(2, 8, 3)),
            ]
            optimizer = torch.optim.Adam([
                {"params": [p], "lr": lr}
                for p, lr in zip(parameters, (1.6e-4, 5e-3, 1e-3, 5e-2, 2.5e-3, 1.25e-4))
            ], eps=1e-15)
            sum(p.sum() for p in parameters).backward()
            optimizer.step()
            means, scales, quats, opac, dc, rest = parameters
            save_checkpoint(Path(tmp), trainer._pack(
                step=1, means=means, scales=scales, quats=quats,
                opac=opac, dc=dc, rest=rest, opt=optimizer,
                cfg=cfg, scene_scale=1.0,
            ))
            sample = {
                "image": torch.ones(8, 8, 3), "alpha": torch.ones(8, 8),
                "K": torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]),
                "viewmat": torch.eye(4), "height": 8, "width": 8,
            }
            args = [
                "run_train.py", "--data", tmp, "--out", tmp, "--steps", "3",
                "--resume", "--cpu-fallback", "--ckpt-every", "1",
            ]
            with (
                patch.object(sys, "argv", args),
                patch.object(trainer, "OpenCLGaussianRasterizer", side_effect=RuntimeError("test")),
                patch.object(trainer, "ColmapRGBADataset", return_value=[sample]),
                patch.object(trainer, "init_gaussians_from_points3d") as initialize,
            ):
                trainer.main()
                initialize.assert_not_called()
                saved = load_checkpoint(Path(tmp) / "ckpt_latest.pt")
                self.assertEqual(saved["step"], 3)
                self.assertFalse(torch.equal(saved["means"], means))
                means_state = next(iter(saved["optimizer"]["state"].values()))
                self.assertEqual(means_state["step"].item(), 3)
                trainer.main()
                args.extend(["--data-factor", "2"])
                with self.assertRaisesRegex(SystemExit, "data_factor"):
                    trainer.main()


if __name__ == "__main__":
    unittest.main()
