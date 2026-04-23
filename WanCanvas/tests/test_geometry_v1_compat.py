from __future__ import annotations

import unittest

from wancanvas.data.contracts import Rect
from wancanvas.data.geometry import normalize_relative_position, relative_position_from_regions


class GeometryV1CompatTest(unittest.TestCase):
    def test_relative_position_keeps_v1_six_fields(self) -> None:
        anchor = Rect(top=100, left=100, height=384, width=384)
        target = Rect(top=180, left=220, height=512, width=512)
        raw = relative_position_from_regions(anchor, target)
        self.assertEqual(len(raw), 6)
        norm = normalize_relative_position(raw, canvas_height=720, canvas_width=1280)
        self.assertEqual(len(norm), 6)
        self.assertTrue(all(isinstance(value, float) for value in norm))


if __name__ == "__main__":
    unittest.main()
