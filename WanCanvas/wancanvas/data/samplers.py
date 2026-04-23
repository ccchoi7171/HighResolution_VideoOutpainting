from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import AnchorTargetPlan, Rect, ResizePlan
from .geometry import relative_position_from_regions
from ..utils.masks import build_binary_mask


@dataclass(slots=True)
class AnchorTargetSamplingConfig:
    target_size: tuple[int, int]
    anchor_size: tuple[int, int] | tuple[int, int, int, int]
    dynamic_anchor_size: bool = False
    overlap_ratio: tuple[float, float, float, float] | None = None
    seed: int | None = None

    def make_rng(self) -> random.Random:
        return random.Random(self.seed)


def _resolve_anchor_size(config: AnchorTargetSamplingConfig, rng: random.Random) -> tuple[int, int]:
    if config.dynamic_anchor_size:
        if len(config.anchor_size) != 4:
            raise ValueError("dynamic anchor_size must have four values: min_h, max_h, min_w, max_w")
        min_h, max_h, min_w, max_w = config.anchor_size
        return rng.randint(min_h, max_h), rng.randint(min_w, max_w)
    if len(config.anchor_size) != 2:
        raise ValueError("static anchor_size must have exactly two values: h, w")
    return int(config.anchor_size[0]), int(config.anchor_size[1])


def sample_anchor_target_plan(
    frame_height: int,
    frame_width: int,
    config: AnchorTargetSamplingConfig,
    *,
    rng: random.Random | None = None,
) -> AnchorTargetPlan:
    rng = rng or config.make_rng()
    target_h, target_w = config.target_size
    anchor_h, anchor_w = _resolve_anchor_size(config, rng)

    resize_plan = None
    if config.overlap_ratio is not None:
        min_oh, max_oh, min_ow, max_ow = config.overlap_ratio
        overlap_h = rng.uniform(min_oh, max_oh)
        overlap_w = rng.uniform(min_ow, max_ow)
        offset_h = int((1.0 - overlap_h) * target_h)
        offset_w = int((1.0 - overlap_w) * target_w)
        required_h = int((offset_h + anchor_h / 2) * 2) + 16
        required_w = int((offset_w + anchor_w / 2) * 2) + 16
        if required_h * 16 / 9 > required_w:
            scale = required_h / frame_height
            new_h = required_h
            new_w = int(frame_width * scale)
        else:
            scale = required_w / frame_width
            new_h = int(frame_height * scale)
            new_w = required_w
        resize_plan = ResizePlan(frame_height, frame_width, new_h, new_w, scale)
        frame_height, frame_width = new_h, new_w
        anchor_center_y = frame_height // 2
        anchor_center_x = frame_width // 2
        target_center_y = (
            anchor_center_y + offset_h + anchor_h // 2 - target_h // 2
            if rng.random() > 0.5
            else anchor_center_y - offset_h - anchor_h // 2 + target_h // 2
        )
        target_center_x = (
            anchor_center_x + offset_w + anchor_w // 2 - target_w // 2
            if rng.random() > 0.5
            else anchor_center_x - offset_w - anchor_w // 2 + target_w // 2
        )
        anchor = Rect(anchor_center_y - anchor_h // 2, anchor_center_x - anchor_w // 2, anchor_h, anchor_w)
        target = Rect(target_center_y - target_h // 2, target_center_x - target_w // 2, target_h, target_w)
        notes = (f"sampled_overlap=({overlap_h:.4f},{overlap_w:.4f})",)
    else:
        max_target_h = max(frame_height - target_h, 0)
        max_target_w = max(frame_width - target_w, 0)
        target = Rect(rng.randint(0, max_target_h), rng.randint(0, max_target_w), target_h, target_w)
        anchor = Rect(frame_height // 2 - anchor_h // 2, frame_width // 2 - anchor_w // 2, anchor_h, anchor_w)
        notes = ()

    intersection = anchor.intersection(target)
    known_region_in_target = intersection.to_local(target) if intersection else None
    relative_position = relative_position_from_regions(anchor, target)
    return AnchorTargetPlan(
        anchor_region=anchor,
        target_region=target,
        known_region_in_target=known_region_in_target,
        relative_position_raw=relative_position,
        resize_plan=resize_plan,
        notes=notes,
    )


def build_known_mask(target_height: int, target_width: int, known_region: Rect | None) -> list[list[int]]:
    return build_binary_mask(target_height, target_width, known_region)
