import tempfile
import unittest
from pathlib import Path

import torch

from engines_common.resume import (
    assert_same_settings,
    load_checkpoint,
    resolve_resume_checkpoint,
    save_checkpoint,
)


class ResumeTests(unittest.TestCase):
    def test_missing_checkpoint_starts_fresh(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            self.assertIsNone(resolve_resume_checkpoint(Path(tmp), True))

    def test_latest_checkpoint_and_numbered_fallback(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            out = Path(tmp)
            for step in (2, 10):
                save_checkpoint(
                    out, {"step": step, "means": torch.ones(2, 3) * step},
                    also_named=out / f"ckpt_{step}.pt",
                )
            latest = resolve_resume_checkpoint(out, True)
            self.assertEqual(latest.name, "ckpt_latest.pt")
            self.assertEqual(load_checkpoint(latest)["step"], 10)
            self.assertIsNone(resolve_resume_checkpoint(out, False))
            latest.unlink()
            fallback = resolve_resume_checkpoint(out, True)
            self.assertEqual(load_checkpoint(fallback)["step"], 10)

    def test_mismatched_settings_are_rejected(self):
        with self.assertRaisesRegex(SystemExit, "data_factor"):
            assert_same_settings(
                {"data_factor": 4}, {"data_factor": 2}, ("data_factor",)
            )

    def test_equivalent_dataset_paths_are_accepted(self):
        assert_same_settings(
            {"data": "."}, {"data": str(Path.cwd())}, ("data",)
        )


if __name__ == "__main__":
    unittest.main()
