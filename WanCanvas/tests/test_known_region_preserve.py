from __future__ import annotations

import unittest

from wancanvas.pipelines.known_region import describe_preserve_action
from wancanvas.utils.masks import build_binary_mask
from wancanvas.data.contracts import Rect


class KnownRegionPreserveTest(unittest.TestCase):
    def test_known_fraction_changes_with_mask_area(self) -> None:
        mask_small = build_binary_mask(8, 8, Rect(0, 0, 1, 1))
        mask_large = build_binary_mask(8, 8, Rect(0, 0, 4, 4))
        action_small = describe_preserve_action(mask_small, mode="overwrite", step_index=0, total_steps=1)
        action_large = describe_preserve_action(mask_large, mode="overwrite", step_index=0, total_steps=1)
        self.assertLess(action_small.preserve_fraction, action_large.preserve_fraction)


if __name__ == "__main__":
    unittest.main()
