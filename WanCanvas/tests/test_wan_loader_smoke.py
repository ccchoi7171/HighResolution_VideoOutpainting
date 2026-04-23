from __future__ import annotations

import unittest

from wancanvas.backbones.wan_loader import WanLoader


class WanLoaderSmokeTest(unittest.TestCase):
    def test_smoke_report_contains_ti2v_defaults(self) -> None:
        report = WanLoader().smoke_validate(download_model=False, strict_runtime=False)
        self.assertEqual(report.model["base_model_id"], "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
        self.assertEqual(report.model["base_pipeline_class"], "WanPipeline")
        self.assertIn("ready_for_download", report.to_dict())


if __name__ == "__main__":
    unittest.main()
