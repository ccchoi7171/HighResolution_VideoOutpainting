from __future__ import annotations

from typing import Iterable

from .contracts import Rect


def relative_position_from_regions(anchor: Rect, target: Rect) -> tuple[int, int, int, int, int, int]:
    anchor_center_y, anchor_center_x = anchor.center
    target_center_y, target_center_x = target.center
    return (
        target_center_y - anchor_center_y,
        target_center_x - anchor_center_x,
        anchor.height,
        anchor.width,
        target.height,
        target.width,
    )


def normalize_relative_position(
    raw: Iterable[int],
    *,
    canvas_height: int,
    canvas_width: int,
    include_canvas: bool = False,
) -> tuple[float, ...]:
    values = list(raw)
    if len(values) != 6:
        raise ValueError("relative_position v1 must contain exactly 6 fields")
    dy, dx, anchor_h, anchor_w, target_h, target_w = values
    norm = (
        dy / max(canvas_height, 1),
        dx / max(canvas_width, 1),
        anchor_h / max(canvas_height, 1),
        anchor_w / max(canvas_width, 1),
        target_h / max(canvas_height, 1),
        target_w / max(canvas_width, 1),
    )
    if include_canvas:
        return norm + (float(canvas_height), float(canvas_width))
    return norm
