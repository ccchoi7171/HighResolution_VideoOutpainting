from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..data.contracts import Rect


@dataclass(frozen=True, slots=True)
class TilePlan:
    row: int
    col: int
    region: Rect


class WindowScheduler:
    def __init__(self, tile_height: int, tile_width: int, overlap_height: int, overlap_width: int) -> None:
        if tile_height <= 0 or tile_width <= 0:
            raise ValueError("tile size must be positive")
        if overlap_height < 0 or overlap_width < 0:
            raise ValueError("overlap must be non-negative")
        if overlap_height >= tile_height or overlap_width >= tile_width:
            raise ValueError("overlap must be smaller than tile size")
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.overlap_height = overlap_height
        self.overlap_width = overlap_width

    @staticmethod
    def _axis_positions(canvas_size: int, tile_size: int, overlap_size: int) -> list[int]:
        if canvas_size < tile_size:
            raise ValueError("canvas must be at least as large as one tile")
        if canvas_size == tile_size:
            return [0]
        step = tile_size - overlap_size
        tail = canvas_size - tile_size
        positions = [0]
        while True:
            next_start = positions[-1] + step
            if next_start >= tail:
                if positions[-1] != tail:
                    positions.append(tail)
                break
            lookahead = next_start + step
            if lookahead >= tail:
                positions.append(tail)
                break
            positions.append(next_start)
        return positions

    def plan_canvas(self, canvas_height: int, canvas_width: int) -> list[TilePlan]:
        row_positions = self._axis_positions(canvas_height, self.tile_height, self.overlap_height)
        col_positions = self._axis_positions(canvas_width, self.tile_width, self.overlap_width)
        tiles: list[TilePlan] = []
        for row, top in enumerate(row_positions):
            for col, left in enumerate(col_positions):
                tiles.append(
                    TilePlan(
                        row=row,
                        col=col,
                        region=Rect(top=top, left=left, height=self.tile_height, width=self.tile_width),
                    )
                )
        return tiles

    def relative_position_for_tile(self, anchor_region: Rect, tile_region: Rect) -> tuple[int, int, int, int, int, int]:
        return (
            tile_region.center[0] - anchor_region.center[0],
            tile_region.center[1] - anchor_region.center[1],
            anchor_region.height,
            anchor_region.width,
            tile_region.height,
            tile_region.width,
        )

    def covered_area(self, tiles: Iterable[TilePlan]) -> tuple[int, int, int, int]:
        tops = [tile.region.top for tile in tiles]
        lefts = [tile.region.left for tile in tiles]
        bottoms = [tile.region.bottom for tile in tiles]
        rights = [tile.region.right for tile in tiles]
        return min(tops), min(lefts), max(bottoms), max(rights)
