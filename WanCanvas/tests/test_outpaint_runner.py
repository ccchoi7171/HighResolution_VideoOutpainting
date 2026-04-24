from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wancanvas.backbones.wan_loader import WanLoader
from wancanvas.inference.outpaint_runner import (
    OutpaintInferenceConfig,
    _ensure_output_dirs,
    _normalize_num_frames,
    _resolve_run_name,
    run_outpaint_inference,
)
from wancanvas.utils.logging import read_json_report

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, 'torch not installed')
class OutpaintRunnerTest(unittest.TestCase):
    def test_run_name_is_stable(self) -> None:
        config = OutpaintInferenceConfig(prompt='Polar bear walking in snow', source_video_tensor=torch.zeros(9, 3, 64, 64))
        self.assertTrue(_resolve_run_name(config).startswith('outpaint-polar-bear'))

    def test_num_frames_are_normalized_to_valid_wan_count(self) -> None:
        self.assertEqual(_normalize_num_frames(7), 5)
        self.assertEqual(_normalize_num_frames(8), 9)
        self.assertEqual(_normalize_num_frames(9), 9)

    def test_output_dirs_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root, original_dir, samples_dir, logs_dir = _ensure_output_dirs(tmpdir, 'demo-run')
            self.assertTrue(run_root.exists())
            self.assertTrue(original_dir.exists())
            self.assertTrue(samples_dir.exists())
            self.assertTrue(logs_dir.exists())

    def test_outpaint_smoke_runner_writes_video_and_metadata(self) -> None:
        runtime = WanLoader().load_pipeline(runtime_variant='smoke', device='cpu', torch_dtype='float32')
        source_video = torch.linspace(0.0, 1.0, 9 * 3 * 64 * 64, dtype=torch.float32).view(9, 3, 64, 64)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = run_outpaint_inference(
                OutpaintInferenceConfig(
                    prompt='extend the snowy panorama',
                    source_video_tensor=source_video,
                    output_root=tmpdir,
                    canvas_height=64,
                    canvas_width=96,
                    num_frames=9,
                    num_inference_steps=2,
                    guidance_scale=2.5,
                    fps=8,
                    tile_height=64,
                    tile_width=64,
                    overlap_height=16,
                    overlap_width=16,
                    rounds=2,
                    runtime_variant='smoke',
                ),
                runtime=runtime,
            )
            self.assertTrue(Path(artifacts.sample_video_path).exists())
            self.assertGreater(Path(artifacts.sample_video_path).stat().st_size, 0)
            metadata = read_json_report(artifacts.metadata_path)
            self.assertEqual(metadata['generation_mode'], 'outpaint')
            self.assertEqual(metadata['model']['runtime_variant'], 'smoke')
            self.assertTrue(metadata['tile_records'])
            self.assertTrue(Path(artifacts.plan_path).exists())


if __name__ == '__main__':
    unittest.main()
