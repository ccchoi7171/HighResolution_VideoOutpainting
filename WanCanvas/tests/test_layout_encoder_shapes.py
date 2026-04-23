from __future__ import annotations

import unittest

from wancanvas.models.layout_encoder import LayoutEncoderConfig, SimpleLayoutEncoder

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class LayoutEncoderTest(unittest.TestCase):
    def test_output_shape_description(self) -> None:
        encoder = SimpleLayoutEncoder(LayoutEncoderConfig())
        self.assertEqual(encoder.describe_output_shape(2), (2, 8, 1024))

    @unittest.skipIf(torch is None, "torch not installed in wancanvas env")
    def test_forward_shape(self) -> None:
        encoder = SimpleLayoutEncoder(LayoutEncoderConfig())
        video = torch.randn(2, 6, 3, 32, 32)
        output = encoder(video)
        self.assertEqual(tuple(output.tokens.shape), (2, 8, 1024))


if __name__ == "__main__":
    unittest.main()
