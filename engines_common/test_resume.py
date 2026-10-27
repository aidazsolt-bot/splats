import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from engines_common.resume import (
    assert_same_settings,
    load_checkpoint,
    resolve_resume_checkpoint,
    save_checkpoint,
)


class ResumeTests(unittest.TestCase):
    def test_failed_serialization_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            latest = save_checkpoint(out, {"step": 1})

            def fail_save(payload, stream):
                stream.write(b"incomplete checkpoint")
                raise OSError("disk write failed")

            with patch("engines_common.resume.torch.save", side_effect=fail_save):
                with self.assertRaisesRegex(OSError, "disk write failed"):
                    save_checkpoint(out, {"step": 2})
            self.assertEqual(load_checkpoint(latest)["step"], 1)
            self.assertEqual(list(out.glob("*.tmp")), [])

    def test_transient_replacement_lock_is_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            latest = save_checkpoint(out, {"step": 1})
            import os

            replace = os.replace
            attempts = 0

            def locked_once(source, target):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError("locked")
                replace(source, target)

            with (
                patch("engines_common.resume.os.replace", side_effect=locked_once),
                patch("engines_common.resume.time.sleep"),
            ):
                save_checkpoint(out, {"step": 2})
            self.assertEqual(attempts, 2)
            self.assertEqual(load_checkpoint(latest)["step"], 2)
            self.assertEqual(list(out.glob("*.tmp")), [])

    def test_persistent_replacement_lock_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            latest = save_checkpoint(out, {"step": 1})
            with (
                patch("engines_common.resume.os.replace", side_effect=PermissionError("locked")),
                patch("engines_common.resume.time.sleep"),
            ):
                with self.assertRaisesRegex(PermissionError, "previous file is unchanged"):
                    save_checkpoint(out, {"step": 2})
            self.assertEqual(load_checkpoint(latest)["step"], 1)
            self.assertEqual(list(out.glob("*.tmp")), [])

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
