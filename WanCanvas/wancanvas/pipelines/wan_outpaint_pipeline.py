from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import math

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..data.contracts import Rect
from ..pipelines.overlap_merge import gaussian_weights_2d
from ..pipelines.size_alignment import SizeAlignmentRule, snap_spatial_size, validate_spatial_size
from ..pipelines.window_scheduler import WindowScheduler
from ..models.wan_outpaint_wrapper import WanForwardRequest, WanOutpaintWrapper
from ..utils.latent_ops import estimate_latent_hw


@dataclass(slots=True)
class OutpaintRequest:
    prompt: str
    frame_count: int
    fps: int
    canvas_height: int
    canvas_width: int
    anchor_region: Rect
    tile_height: int
    tile_width: int
    overlap_height: int
    overlap_width: int
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MultiRoundOutpaintRequest:
    prompt: str
    frame_count: int
    fps: int
    final_canvas_height: int
    final_canvas_width: int
    anchor_region: Rect
    tile_height: int
    tile_width: int
    overlap_height: int
    overlap_width: int
    rounds: int = 1
    extras: dict[str, Any] = field(default_factory=dict)


class WanOutpaintPipeline:
    def __init__(self, wrapper: WanOutpaintWrapper, *, size_rule: SizeAlignmentRule) -> None:
        self.wrapper = wrapper
        self.size_rule = size_rule

    def _validate_aligned_size(self, height: int, width: int) -> None:
        ok, errors = validate_spatial_size(height, width, self.size_rule)
        if not ok:
            raise ValueError('size alignment failed: ' + '; '.join(errors))

    @staticmethod
    def _covered_bounds_from_rects(rects: list[Rect]) -> tuple[int, int, int, int]:
        return min(rect.top for rect in rects), min(rect.left for rect in rects), max(rect.bottom for rect in rects), max(rect.right for rect in rects)

    @staticmethod
    def _dry_run_request_tensors(*, frame_count: int, target_height: int, target_width: int) -> tuple[Any, Any, Any]:
        if torch is None:
            return 'latent-dry-run', 'scheduler-dry-run', 'text-dry-run'
        latent_hw = estimate_latent_hw(target_height, target_width)
        return (
            torch.zeros((1, frame_count, 16, latent_hw[0], latent_hw[1]), dtype=torch.float32),
            torch.tensor([999], dtype=torch.int64),
            torch.zeros((1, 1, 1024), dtype=torch.float32),
        )

    def plan_request(self, request: OutpaintRequest) -> dict[str, Any]:
        self._validate_aligned_size(request.canvas_height, request.canvas_width)
        self._validate_aligned_size(request.tile_height, request.tile_width)
        scheduler = WindowScheduler(
            tile_height=request.tile_height,
            tile_width=request.tile_width,
            overlap_height=request.overlap_height,
            overlap_width=request.overlap_width,
        )
        tiles = scheduler.plan_canvas(request.canvas_height, request.canvas_width)
        tile_payload = []
        for tile in tiles:
            tile_payload.append(
                {
                    'row': tile.row,
                    'col': tile.col,
                    'region': asdict(tile.region),
                    'relative_position_raw': scheduler.relative_position_for_tile(request.anchor_region, tile.region),
                }
            )
        return {
            'prompt': request.prompt,
            'frame_count': request.frame_count,
            'fps': request.fps,
            'anchor_region': asdict(request.anchor_region),
            'tile_count': len(tile_payload),
            'tiles': tile_payload,
            'extras': dict(request.extras),
        }

    @staticmethod
    def _allocate_alignment_delta(current_start: int, current_end: int, final_end: int, delta: int) -> tuple[int, int]:
        if delta <= 0:
            return current_start, current_end
        end_slack = final_end - current_end
        add_end = min(delta, end_slack)
        current_end += add_end
        delta -= add_end
        current_start -= delta
        if current_start < 0:
            raise ValueError('alignment adjustment pushed canvas bounds negative')
        return current_start, current_end

    def _round_canvas_region(self, request: MultiRoundOutpaintRequest, round_index: int) -> tuple[Rect, Rect]:
        margins = {
            'top': request.anchor_region.top,
            'left': request.anchor_region.left,
            'bottom': request.final_canvas_height - request.anchor_region.bottom,
            'right': request.final_canvas_width - request.anchor_region.right,
        }
        top_expand = math.ceil(margins['top'] * round_index / request.rounds)
        left_expand = math.ceil(margins['left'] * round_index / request.rounds)
        bottom_expand = math.ceil(margins['bottom'] * round_index / request.rounds)
        right_expand = math.ceil(margins['right'] * round_index / request.rounds)

        current_top = request.anchor_region.top - top_expand
        current_left = request.anchor_region.left - left_expand
        current_bottom = request.anchor_region.bottom + bottom_expand
        current_right = request.anchor_region.right + right_expand

        current_height = current_bottom - current_top
        current_width = current_right - current_left
        aligned_height, aligned_width = snap_spatial_size(current_height, current_width, self.size_rule, mode='ceil')
        current_top, current_bottom = self._allocate_alignment_delta(current_top, current_bottom, request.final_canvas_height, aligned_height - current_height)
        current_left, current_right = self._allocate_alignment_delta(current_left, current_right, request.final_canvas_width, aligned_width - current_width)

        canvas_region = Rect(
            top=current_top,
            left=current_left,
            height=current_bottom - current_top,
            width=current_right - current_left,
        )
        anchor_local = Rect(
            top=request.anchor_region.top - canvas_region.top,
            left=request.anchor_region.left - canvas_region.left,
            height=request.anchor_region.height,
            width=request.anchor_region.width,
        )
        return canvas_region, anchor_local

    def plan_multi_round_request(self, request: MultiRoundOutpaintRequest) -> dict[str, Any]:
        if request.rounds <= 0:
            raise ValueError('rounds must be positive')
        self._validate_aligned_size(request.final_canvas_height, request.final_canvas_width)
        self._validate_aligned_size(request.tile_height, request.tile_width)

        round_payloads: list[dict[str, Any]] = []
        canvas_heights: list[int] = []
        canvas_widths: list[int] = []
        anchor_inside_ok = True
        relative_positions_ok = True
        tile_counts_ok = True
        local_coverage_ok = True
        global_coverage_ok = True
        kernel_center_gt_corner_ok = True

        for round_index in range(1, request.rounds + 1):
            canvas_region, anchor_local = self._round_canvas_region(request, round_index)
            effective_tile_height = min(request.tile_height, canvas_region.height)
            effective_tile_width = min(request.tile_width, canvas_region.width)
            effective_overlap_height = min(request.overlap_height, max(effective_tile_height - 1, 0))
            effective_overlap_width = min(request.overlap_width, max(effective_tile_width - 1, 0))
            scheduler = WindowScheduler(
                tile_height=effective_tile_height,
                tile_width=effective_tile_width,
                overlap_height=effective_overlap_height,
                overlap_width=effective_overlap_width,
            )
            kernel = gaussian_weights_2d(effective_tile_width, effective_tile_height)
            center_weight = kernel[effective_tile_height // 2][effective_tile_width // 2]
            corner_weight = kernel[0][0]
            kernel_center_gt_corner_ok = kernel_center_gt_corner_ok and center_weight > corner_weight
            tiles = scheduler.plan_canvas(canvas_region.height, canvas_region.width)
            local_coverage = scheduler.covered_area(tiles)
            global_regions: list[Rect] = []
            tile_payloads: list[dict[str, Any]] = []
            for tile in tiles:
                global_region = Rect(
                    top=canvas_region.top + tile.region.top,
                    left=canvas_region.left + tile.region.left,
                    height=tile.region.height,
                    width=tile.region.width,
                )
                global_regions.append(global_region)
                relative_position = scheduler.relative_position_for_tile(anchor_local, tile.region)
                relative_positions_ok = relative_positions_ok and len(relative_position) == 6
                tile_payloads.append(
                    {
                        'row': tile.row,
                        'col': tile.col,
                        'region_local': asdict(tile.region),
                        'region_global': asdict(global_region),
                        'relative_position_raw': relative_position,
                    }
                )

            global_coverage = self._covered_bounds_from_rects(global_regions)
            expected_local_coverage = (0, 0, canvas_region.height, canvas_region.width)
            expected_global_coverage = (canvas_region.top, canvas_region.left, canvas_region.bottom, canvas_region.right)
            local_coverage_ok = local_coverage_ok and local_coverage == expected_local_coverage
            global_coverage_ok = global_coverage_ok and global_coverage == expected_global_coverage
            anchor_inside_ok = anchor_inside_ok and canvas_region.intersection(request.anchor_region) == request.anchor_region
            tile_counts_ok = tile_counts_ok and len(tiles) > 0
            canvas_heights.append(canvas_region.height)
            canvas_widths.append(canvas_region.width)
            round_payloads.append(
                {
                    'round_index': round_index,
                    'canvas_region_global': asdict(canvas_region),
                    'canvas_size': [canvas_region.height, canvas_region.width],
                    'anchor_region_local': asdict(anchor_local),
                    'anchor_region_global': asdict(request.anchor_region),
                    'tile_count': len(tile_payloads),
                    'tiles': tile_payloads,
                    'coverage_local': list(local_coverage),
                    'coverage_global': list(global_coverage),
                    'merge_kernel': {
                        'shape': [effective_tile_height, effective_tile_width],
                        'center_weight': center_weight,
                        'corner_weight': corner_weight,
                        'strategy': 'gaussian',
                    },
                }
            )

        invariants = {
            'round_count_matches': len(round_payloads) == request.rounds,
            'canvas_heights_non_decreasing': all(canvas_heights[idx] <= canvas_heights[idx + 1] for idx in range(len(canvas_heights) - 1)),
            'canvas_widths_non_decreasing': all(canvas_widths[idx] <= canvas_widths[idx + 1] for idx in range(len(canvas_widths) - 1)),
            'final_canvas_matches_request': round_payloads[-1]['canvas_size'] == [request.final_canvas_height, request.final_canvas_width],
            'tile_counts_positive': tile_counts_ok,
            'anchor_inside_every_round': anchor_inside_ok,
            'relative_positions_v1_compatible': relative_positions_ok,
            'local_coverage_complete': local_coverage_ok,
            'global_coverage_complete': global_coverage_ok,
            'merge_kernel_center_gt_corner': kernel_center_gt_corner_ok,
        }
        return {
            'prompt': request.prompt,
            'frame_count': request.frame_count,
            'fps': request.fps,
            'final_canvas': [request.final_canvas_height, request.final_canvas_width],
            'anchor_region_global': asdict(request.anchor_region),
            'rounds': round_payloads,
            'invariants': invariants,
            'extras': dict(request.extras),
        }

    def dry_run(self, request: OutpaintRequest) -> dict[str, Any]:
        plan = self.plan_request(request)
        noisy_latents, timesteps, prompt_embeds = self._dry_run_request_tensors(
            frame_count=request.frame_count,
            target_height=request.tile_height,
            target_width=request.tile_width,
        )
        prepared = self.wrapper.dry_run(
            WanForwardRequest(
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                known_region_state={
                    'mode': 'overwrite',
                    'canvas_height': request.canvas_height,
                    'canvas_width': request.canvas_width,
                    'anchor_region': asdict(request.anchor_region),
                    'target_region': None,
                    'scope': 'pipeline-dry-run',
                },
                extras={
                    'tile_count': plan['tile_count'],
                    'canvas_height': request.canvas_height,
                    'canvas_width': request.canvas_width,
                    'frame_count': request.frame_count,
                    'dry_run_text_token_count': 1,
                    'dry_run_token_dim': 1024,
                },
            )
        )
        return {'plan': plan, 'wrapper': prepared}

    def dry_run_multi_round(self, request: MultiRoundOutpaintRequest) -> dict[str, Any]:
        plan = self.plan_multi_round_request(request)
        noisy_latents, timesteps, prompt_embeds = self._dry_run_request_tensors(
            frame_count=request.frame_count,
            target_height=request.tile_height,
            target_width=request.tile_width,
        )
        prepared = self.wrapper.dry_run(
            WanForwardRequest(
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                known_region_state={
                    'mode': 'overwrite',
                    'canvas_height': request.final_canvas_height,
                    'canvas_width': request.final_canvas_width,
                    'anchor_region': asdict(request.anchor_region),
                    'target_region': None,
                    'scope': 'multi-round-pipeline-dry-run',
                },
                extras={
                    'round_count': len(plan['rounds']),
                    'tile_counts_by_round': [round_plan['tile_count'] for round_plan in plan['rounds']],
                    'final_canvas_height': request.final_canvas_height,
                    'final_canvas_width': request.final_canvas_width,
                    'frame_count': request.frame_count,
                    'dry_run_text_token_count': 1,
                    'dry_run_token_dim': 1024,
                },
            )
        )
        return {'plan': plan, 'wrapper': prepared}
