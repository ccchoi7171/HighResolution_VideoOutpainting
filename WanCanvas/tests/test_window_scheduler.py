from __future__ import annotations

import unittest

from wancanvas.data.contracts import Rect
from wancanvas.pipelines.window_scheduler import WindowScheduler


class WindowSchedulerTest(unittest.TestCase):
    def test_tiles_cover_canvas(self) -> None:
        scheduler = WindowScheduler(tile_height=720, tile_width=720, overlap_height=176, overlap_width=176)
        tiles = scheduler.plan_canvas(720, 1280)
        top, left, bottom, right = scheduler.covered_area(tiles)
        self.assertEqual((top, left), (0, 0))
        self.assertEqual((bottom, right), (720, 1280))
        self.assertEqual(len(tiles), 2)

    def test_axis_positions_deduplicate_near_duplicate_tail_tile(self) -> None:
        positions = WindowScheduler._axis_positions(canvas_size=1280, tile_size=720, overlap_size=176)
        self.assertEqual(positions, [0, 560])

    def test_axis_positions_use_tail_once_for_intermediate_canvas(self) -> None:
        positions = WindowScheduler._axis_positions(canvas_size=992, tile_size=720, overlap_size=176)
        self.assertEqual(positions, [0, 272])

    def test_relative_position_is_v1_compatible(self) -> None:
        scheduler = WindowScheduler(tile_height=720, tile_width=720, overlap_height=176, overlap_width=176)
        raw = scheduler.relative_position_for_tile(Rect(100, 100, 384, 384), Rect(0, 0, 720, 720))
        self.assertEqual(len(raw), 6)


if __name__ == "__main__":
    unittest.main()
