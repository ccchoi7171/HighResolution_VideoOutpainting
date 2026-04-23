from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .contracts import CanvasMeta, FYCOutpaintSample, Rect
from .geometry import normalize_relative_position
from .samplers import AnchorTargetSamplingConfig, build_known_mask, sample_anchor_target_plan


@dataclass(slots=True)
class DatasetRecord:
    source_id: str
    prompt: str
    frame_height: int
    frame_width: int
    frame_count: int = 16
    fps: int = 16
    payload: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CropReference:
    source_id: str
    region: Rect
    frame_count: int


class WanCanvasDataset(Sequence[FYCOutpaintSample]):
    def __init__(
        self,
        records: Sequence[DatasetRecord],
        sampling_config: AnchorTargetSamplingConfig,
        *,
        frame_loader: Callable[[DatasetRecord], Any] | None = None,
        cropper: Callable[[Any, Rect], Any] | None = None,
    ) -> None:
        self.records = list(records)
        self.sampling_config = sampling_config
        self.frame_loader = frame_loader
        self.cropper = cropper

    def __len__(self) -> int:
        return len(self.records)

    def _crop_or_reference(self, record: DatasetRecord, region: Rect) -> Any:
        if self.frame_loader is None or self.cropper is None:
            return CropReference(record.source_id, region, record.frame_count)
        frames = self.frame_loader(record)
        return self.cropper(frames, region)

    def __getitem__(self, index: int) -> FYCOutpaintSample:
        record = self.records[index]
        plan = sample_anchor_target_plan(record.frame_height, record.frame_width, self.sampling_config)
        known_mask = build_known_mask(plan.target_region.height, plan.target_region.width, plan.known_region_in_target)
        canvas_meta = CanvasMeta(
            canvas_height=record.frame_height,
            canvas_width=record.frame_width,
            source_id=record.source_id,
            anchor_region=plan.anchor_region,
            target_region=plan.target_region,
            resize_plan=plan.resize_plan,
            extras=dict(record.extras),
        )
        return FYCOutpaintSample(
            anchor_video=self._crop_or_reference(record, plan.anchor_region),
            target_video=self._crop_or_reference(record, plan.target_region),
            known_mask=known_mask,
            relative_position_raw=plan.relative_position_raw,
            relative_position_norm=normalize_relative_position(
                plan.relative_position_raw,
                canvas_height=record.frame_height,
                canvas_width=record.frame_width,
            ),
            prompt=record.prompt,
            fps=record.fps,
            frame_count=record.frame_count,
            canvas_meta=canvas_meta,
        )
