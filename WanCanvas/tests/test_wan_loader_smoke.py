from __future__ import annotations

import unittest

from wancanvas.backbones.wan_loader import WanLoader


class WanLoaderSmokeTest(unittest.TestCase):
    def test_smoke_report_contains_i2v_defaults(self) -> None:
        report = WanLoader().smoke_validate(download_model=False, strict_runtime=False)
        self.assertEqual(report.model['base_model_id'], 'Wan-AI/Wan2.2-TI2V-5B-Diffusers')
        self.assertEqual(report.model['base_pipeline_class'], 'WanImageToVideoPipeline')
        self.assertIn('ready_for_download', report.to_dict())

    def test_smoke_runtime_bundle_is_available_without_weights(self) -> None:
        bundle = WanLoader().load_pipeline(runtime_variant='smoke', device='cpu', torch_dtype='float32')
        summary = bundle.summary()
        self.assertEqual(summary['runtime_variant'], 'smoke')
        self.assertIn('smoke', summary['pipeline_class'])
        self.assertTrue(hasattr(bundle.pipeline, 'transformer'))
        self.assertTrue(hasattr(bundle.pipeline, 'scheduler'))


if __name__ == '__main__':
    unittest.main()
