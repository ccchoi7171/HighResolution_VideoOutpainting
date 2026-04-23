from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Rect:
    top: int
    left: int
    height: int
    width: int

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def center(self) -> tuple[int, int]:
        return (self.top + self.height // 2, self.left + self.width // 2)

    def intersection(self, other: "Rect") -> "Rect | None":
        top = max(self.top, other.top)
        left = max(self.left, other.left)
        bottom = min(self.bottom, other.bottom)
        right = min(self.right, other.right)
        if bottom <= top or right <= left:
            return None
        return Rect(top=top, left=left, height=bottom - top, width=right - left)

    def to_local(self, parent: "Rect") -> "Rect":
        return Rect(
            top=self.top - parent.top,
            left=self.left - parent.left,
            height=self.height,
            width=self.width,
        )


@dataclass(frozen=True, slots=True)
class ResizePlan:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    scale_factor: float


@dataclass(frozen=True, slots=True)
class AnchorTargetPlan:
    anchor_region: Rect
    target_region: Rect
    known_region_in_target: Rect | None
    relative_position_raw: tuple[int, int, int, int, int, int]
    resize_plan: ResizePlan | None = None
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class CanvasMeta:
    canvas_height: int
    canvas_width: int
    round_index: int = 0
    tile_index: int | None = None
    source_id: str | None = None
    anchor_region: Rect | None = None
    target_region: Rect | None = None
    resize_plan: ResizePlan | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FYCOutpaintSample:
    anchor_video: Any
    target_video: Any
    known_mask: Any
    relative_position_raw: tuple[int, int, int, int, int, int]
    relative_position_norm: tuple[float, ...]
    prompt: str
    fps: int
    frame_count: int
    canvas_meta: CanvasMeta
