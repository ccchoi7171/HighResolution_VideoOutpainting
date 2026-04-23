from __future__ import annotations

import unittest

from wancanvas.data.contracts import Rect
from wancanvas.pipelines.known_region import describe_preserve_action
from wancanvas.utils.masks import build_binary_mask, validate_binary_mask


class MaskContractTest(unittest.TestCase):
    def test_mask_uses_zero_for_known_and_one_for_generate(self) -> None:
        mask = build_binary_mask(6, 6, Rect(top=1, left=1, height=2, width=3))
        validate_binary_mask(mask)
        self.assertEqual(mask[1][1], 0)
        self.assertEqual(mask[0][0], 1)

    def test_preserve_description_tracks_known_fraction(self) -> None:
        mask = build_binary_mask(4, 4, Rect(top=0, left=0, height=2, width=2))
        action = describe_preserve_action(mask, mode="blend", step_index=1, total_steps=4)
        self.assertGreater(action.preserve_fraction, 0)
        self.assertLessEqual(action.blend_alpha, 1.0)


if __name__ == "__main__":
    unittest.main()
