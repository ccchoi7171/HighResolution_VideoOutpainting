from __future__ import annotations

from typing import Any, Iterable


def build_binary_mask(height: int, width: int, known_region: Any | None) -> list[list[int]]:
    mask = [[1 for _ in range(width)] for _ in range(height)]
    if known_region is None:
        return mask
    top = int(getattr(known_region, "top", 0))
    left = int(getattr(known_region, "left", 0))
    bottom = int(getattr(known_region, "bottom", top))
    right = int(getattr(known_region, "right", left))
    for row in range(max(top, 0), min(bottom, height)):
        for col in range(max(left, 0), min(right, width)):
            mask[row][col] = 0
    return mask


def validate_binary_mask(mask: Iterable[Iterable[int]]) -> None:
    rows = [list(row) for row in mask]
    if not rows or not rows[0]:
        raise ValueError("mask must be a non-empty 2D structure")
    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError("mask rows must all have the same width")
        for value in row:
            if value not in {0, 1}:
                raise ValueError("mask values must be 0 or 1")
