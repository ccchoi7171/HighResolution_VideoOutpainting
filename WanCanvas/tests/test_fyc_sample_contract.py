from __future__ import annotations

import unittest

from wancanvas.data.outpaint_dataset import DatasetRecord, WanCanvasDataset
from wancanvas.data.samplers import AnchorTargetSamplingConfig


class FYCOutpaintSampleTest(unittest.TestCase):
    def test_dataset_returns_fyc_contract(self) -> None:
        dataset = WanCanvasDataset(
            records=[DatasetRecord(source_id="sample", prompt="expand", frame_height=720, frame_width=1280)],
            sampling_config=AnchorTargetSamplingConfig(target_size=(512, 512), anchor_size=(384, 384), seed=13),
        )
        sample = dataset[0]
        self.assertEqual(len(sample.relative_position_raw), 6)
        self.assertEqual(sample.prompt, "expand")
        self.assertEqual(len(sample.known_mask), 512)
        self.assertEqual(len(sample.known_mask[0]), 512)


if __name__ == "__main__":
    unittest.main()
