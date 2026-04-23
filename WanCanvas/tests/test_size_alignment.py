from __future__ import annotations

import unittest

from wancanvas.pipelines.size_alignment import SizeAlignmentRule, snap_spatial_size, validate_spatial_size


class SizeAlignmentTest(unittest.TestCase):
    def test_valid_size(self) -> None:
        ok, errors = validate_spatial_size(720, 1280, SizeAlignmentRule(8, 2))
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_invalid_size(self) -> None:
        ok, errors = validate_spatial_size(721, 1280, SizeAlignmentRule(8, 2))
        self.assertFalse(ok)
        self.assertTrue(errors)

    def test_snap_size(self) -> None:
        self.assertEqual(snap_spatial_size(721, 1279, SizeAlignmentRule(8, 2), mode="ceil"), (736, 1280))


if __name__ == "__main__":
    unittest.main()
