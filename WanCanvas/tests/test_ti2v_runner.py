from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wancanvas.inference.ti2v_runner import Ti2VInferenceConfig, _ensure_output_dirs, _normalize_num_frames, _resolve_run_name


class Ti2VRunnerTest(unittest.TestCase):
    def test_run_name_is_stable(self) -> None:
        config = Ti2VInferenceConfig(prompt="Polar bear walking in snow", source_image=None)
        self.assertTrue(_resolve_run_name(config).startswith("ti2v-polar-bear"))

    def test_num_frames_are_normalized_to_valid_wan_count(self) -> None:
        self.assertEqual(_normalize_num_frames(7), 5)
        self.assertEqual(_normalize_num_frames(8), 9)
        self.assertEqual(_normalize_num_frames(9), 9)

    def test_output_dirs_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root, original_dir, samples_dir, logs_dir = _ensure_output_dirs(tmpdir, "demo-run")
            self.assertTrue(run_root.exists())
            self.assertTrue(original_dir.exists())
            self.assertTrue(samples_dir.exists())
            self.assertTrue(logs_dir.exists())


if __name__ == "__main__":
    unittest.main()
