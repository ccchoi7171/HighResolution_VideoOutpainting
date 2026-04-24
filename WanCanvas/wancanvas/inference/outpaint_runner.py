from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import shutil

import numpy as np
import imageio.v2 as imageio

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..backbones.wan_loader import WanLoader
from ..data.contracts import Rect
from ..data.geometry import normalize_relative_position
from ..models.fyc_conditioning import FYCConditioningBuilder
from ..models.wan_outpaint_wrapper import WanForwardRequest, WanOutpaintWrapper
from ..pipelines.overlap_merge import gaussian_weights_2d
from ..pipelines.size_alignment import SizeAlignmentRule, validate_spatial_size
from ..pipelines.wan_outpaint_pipeline import MultiRoundOutpaintRequest, WanOutpaintPipeline
from ..utils.logging import write_json_report

DEFAULT_NEGATIVE_PROMPT = (
    'overexposed, static shot, blurry details, subtitles, watermark, worst quality, low quality, '
    'jpeg artifacts, malformed anatomy, extra limbs, fused fingers, messy background'
)


@dataclass(slots=True)
class OutpaintInferenceConfig:
    prompt: str
    source_video: str | None = None
    source_video_tensor: Any = None
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    output_root: str = 'runs/wancanvas'
    run_name: str | None = None
    model_id: str = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
    pipeline_class: str = 'WanImageToVideoPipeline'
    cache_dir: str = '.cache/wancanvas'
    device: str = 'cuda'
    torch_dtype: str = 'bfloat16'
    canvas_height: int = 704
    canvas_width: int = 1280
    num_frames: int = 49
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    fps: int = 16
    seed: int = 7
    flow_shift: float = 5.0
    scheduler_name: str | None = 'FlowMatchEulerDiscreteScheduler'
    tile_height: int = 720
    tile_width: int = 720
    overlap_height: int = 176
    overlap_width: int = 176
    rounds: int = 1
    anchor_top: int | None = None
    anchor_left: int | None = None
    anchor_height: int | None = None
    anchor_width: int | None = None
    local_files_only: bool = False
    enable_model_cpu_offload: bool = False
    runtime_variant: str = 'auto'

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError('prompt must not be empty')
        if self.source_video is None and self.source_video_tensor is None:
            raise ValueError('source_video or source_video_tensor must be provided')
        if self.canvas_height <= 0 or self.canvas_width <= 0:
            raise ValueError('canvas_height and canvas_width must be positive')
        if self.num_frames <= 0 or self.num_inference_steps <= 0 or self.fps <= 0:
            raise ValueError('num_frames, num_inference_steps, and fps must be positive')
        if self.runtime_variant not in {'auto', 'pretrained', 'smoke'}:
            raise ValueError('runtime_variant must be auto, pretrained, or smoke')


@dataclass(slots=True)
class OutpaintRunArtifacts:
    run_root: str
    original_dir: str
    samples_dir: str
    logs_dir: str
    prompt_path: str
    negative_prompt_path: str
    metadata_path: str
    sample_video_path: str
    source_video_archive_path: str | None = None
    plan_path: str | None = None
    generation_mode: str = 'outpaint'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TileExecutionRecord:
    round_index: int
    tile_index: int
    region_global: dict[str, int]
    known_fraction: float


def _slugify(text: str, max_len: int = 48) -> str:
    safe: list[str] = []
    for char in text.lower():
        if char.isalnum():
            safe.append(char)
        elif char in {' ', '-', '_'}:
            safe.append('-')
    slug = ''.join(safe).strip('-') or 'wan-outpaint'
    while '--' in slug:
        slug = slug.replace('--', '-')
    return slug[:max_len].rstrip('-') or 'wan-outpaint'


def _normalize_num_frames(num_frames: int) -> int:
    if num_frames <= 0:
        raise ValueError('num_frames must be positive')
    lower = ((num_frames - 1) // 4) * 4 + 1
    upper = lower + 4
    if num_frames == lower:
        return lower
    if abs(num_frames - lower) <= abs(upper - num_frames):
        return max(lower, 1)
    return upper


def _resolve_run_name(config: OutpaintInferenceConfig) -> str:
    if config.run_name:
        return config.run_name
    return f'outpaint-{_slugify(config.prompt)}-s{config.seed}'


def _ensure_output_dirs(output_root: str, run_name: str) -> tuple[Path, Path, Path, Path]:
    run_root = Path(output_root) / run_name
    original_dir = run_root / 'original'
    samples_dir = run_root / 'samples'
    logs_dir = run_root / 'logs'
    for path in (original_dir, samples_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return run_root, original_dir, samples_dir, logs_dir


def _archive_source_video(config: OutpaintInferenceConfig, original_dir: Path) -> Path | None:
    if config.source_video is None:
        return None
    src = Path(config.source_video)
    if not src.exists():
        raise FileNotFoundError(f'source video not found: {src}')
    destination = original_dir / src.name
    shutil.copy2(src, destination)
    return destination


def _read_video_tensor(path: str, *, num_frames: int) -> 'torch.Tensor':
    if torch is None:
        raise RuntimeError('torch is required for outpaint inference')
    reader = imageio.get_reader(path)
    frames: list[np.ndarray] = []
    try:
        for frame in reader:
            frames.append(np.asarray(frame))
            if len(frames) >= num_frames:
                break
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f'No frames were read from {path}')
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    array = np.stack(frames[:num_frames], axis=0)
    return torch.from_numpy(array).float().permute(0, 3, 1, 2) / 255.0


def _write_video_tensor(video: 'torch.Tensor', output_path: Path, *, fps: int) -> Path:
    array = (video.clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for frame in array:
            writer.append_data(frame)
    finally:
        writer.close()
    return output_path


def _resolve_anchor_region(config: OutpaintInferenceConfig, source_video: 'torch.Tensor') -> Rect:
    source_height = int(source_video.shape[-2])
    source_width = int(source_video.shape[-1])
    anchor_height = config.anchor_height or source_height
    anchor_width = config.anchor_width or source_width
    anchor_top = config.anchor_top if config.anchor_top is not None else (config.canvas_height - anchor_height) // 2
    anchor_left = config.anchor_left if config.anchor_left is not None else (config.canvas_width - anchor_width) // 2
    return Rect(top=anchor_top, left=anchor_left, height=anchor_height, width=anchor_width)


def _tile_weight(tile_height: int, tile_width: int, *, device: 'torch.device') -> 'torch.Tensor':
    weights = torch.tensor(gaussian_weights_2d(tile_width, tile_height), dtype=torch.float32, device=device)
    return weights.unsqueeze(0).unsqueeze(0)


def run_outpaint_inference(config: OutpaintInferenceConfig, *, runtime: Any = None) -> OutpaintRunArtifacts:
    if torch is None:
        raise RuntimeError('torch is required for outpaint inference')
    config.validate()
    normalized_num_frames = _normalize_num_frames(config.num_frames)
    rule = SizeAlignmentRule()
    aligned, errors = validate_spatial_size(config.canvas_height, config.canvas_width, rule)
    if not aligned:
        raise ValueError('size alignment failed: ' + '; '.join(errors))
    run_name = _resolve_run_name(config)
    run_root, original_dir, samples_dir, logs_dir = _ensure_output_dirs(config.output_root, run_name)

    prompt_path = original_dir / 'prompt.txt'
    prompt_path.write_text(config.prompt + '\n', encoding='utf-8')
    negative_prompt_path = original_dir / 'negative_prompt.txt'
    negative_prompt_path.write_text(config.negative_prompt + '\n', encoding='utf-8')
    source_video_archive_path = _archive_source_video(config, original_dir)

    if config.source_video_tensor is not None:
        source_video = config.source_video_tensor.float()
    else:
        source_video = _read_video_tensor(config.source_video, num_frames=normalized_num_frames)
    if source_video.ndim != 4:
        raise ValueError('source video tensor must be [F,C,H,W]')
    anchor_region = _resolve_anchor_region(config, source_video)
    if anchor_region.height != source_video.shape[-2] or anchor_region.width != source_video.shape[-1]:
        raise ValueError('anchor region must match the provided source video size')

    wrapper = WanOutpaintWrapper(WanLoader())
    pipeline = WanOutpaintPipeline(wrapper=wrapper, size_rule=rule)
    plan = pipeline.plan_multi_round_request(
        MultiRoundOutpaintRequest(
            prompt=config.prompt,
            frame_count=normalized_num_frames,
            fps=config.fps,
            final_canvas_height=config.canvas_height,
            final_canvas_width=config.canvas_width,
            anchor_region=anchor_region,
            tile_height=config.tile_height,
            tile_width=config.tile_width,
            overlap_height=config.overlap_height,
            overlap_width=config.overlap_width,
            rounds=config.rounds,
        )
    )
    plan_path = write_json_report(logs_dir / 'wan_outpaint_plan.json', plan)

    loaded = runtime or wrapper.load_runtime(
        model_id=config.model_id,
        pipeline_class_name=config.pipeline_class,
        device=config.device,
        torch_dtype=config.torch_dtype,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
        flow_shift=config.flow_shift,
        scheduler_name=config.scheduler_name,
        enable_model_cpu_offload=config.enable_model_cpu_offload,
        runtime_variant=config.runtime_variant,
    )
    pipe = loaded.pipeline if hasattr(loaded, 'pipeline') else loaded
    torch.manual_seed(config.seed)

    working_canvas = torch.zeros(
        normalized_num_frames,
        source_video.shape[1],
        config.canvas_height,
        config.canvas_width,
        dtype=source_video.dtype,
        device=pipe._execution_device,
    )
    working_known = torch.ones(
        1,
        normalized_num_frames,
        1,
        config.canvas_height,
        config.canvas_width,
        dtype=torch.float32,
        device=pipe._execution_device,
    )
    source_video = source_video.to(device=pipe._execution_device)
    working_canvas[:, :, anchor_region.top:anchor_region.bottom, anchor_region.left:anchor_region.right] = source_video
    working_known[:, :, :, anchor_region.top:anchor_region.bottom, anchor_region.left:anchor_region.right] = 0.0

    conditioning_builder = FYCConditioningBuilder()
    tile_records: list[dict[str, Any]] = []
    started = perf_counter()
    for round_plan in plan['rounds']:
        round_accum = torch.zeros_like(working_canvas)
        round_weights = torch.zeros((1, 1, config.canvas_height, config.canvas_width), dtype=torch.float32, device=working_canvas.device)
        for tile_index, tile in enumerate(round_plan['tiles']):
            region = tile['region_global']
            top = region['top']
            left = region['left']
            height = region['height']
            width = region['width']
            condition_video = working_canvas[:, :, top:top + height, left:left + width].unsqueeze(0)
            known_mask = working_known[:, :, :, top:top + height, left:left + width]
            relative_position = torch.tensor(
                [
                    list(
                        normalize_relative_position(
                            tile['relative_position_raw'],
                            canvas_height=config.canvas_height,
                            canvas_width=config.canvas_width,
                        )
                    )
                ],
                dtype=torch.float32,
                device=working_canvas.device,
            )
            conditioning = conditioning_builder.encode(
                anchor_video=source_video.unsqueeze(0),
                relative_position=relative_position,
                known_mask=known_mask,
                prompt_embeds=None,
            )
            request = WanForwardRequest(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                layout_tokens=conditioning.layout.tokens if conditioning.layout is not None else None,
                geometry_tokens=conditioning.geometry.tokens if conditioning.geometry is not None else None,
                mask_tokens=conditioning.mask.tokens if conditioning.mask is not None else None,
                condition_video=condition_video,
                known_mask=known_mask,
                known_region_state={
                    'mode': 'overwrite',
                    'canvas_height': config.canvas_height,
                    'canvas_width': config.canvas_width,
                    'anchor_region': asdict(anchor_region),
                    'target_region': region,
                    'round_index': round_plan['round_index'],
                    'tile_index': tile_index,
                },
                extras={
                    'prompt': config.prompt,
                    'negative_prompt': config.negative_prompt,
                    'frame_count': normalized_num_frames,
                    'target_hw': [height, width],
                    'aligned_target_hw': [height, width],
                },
            )
            generated = wrapper.generate(
                request,
                runtime=pipe,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                output_type='tensor',
            )['frames'][0]
            weights = _tile_weight(height, width, device=working_canvas.device)
            generate_mask = known_mask[0, :, 0].unsqueeze(1)
            spatial_generate_mask = generate_mask.mean(dim=0, keepdim=True)
            round_accum[:, :, top:top + height, left:left + width] += generated * weights * generate_mask
            round_weights[:, :, top:top + height, left:left + width] += weights * spatial_generate_mask
            tile_records.append(
                asdict(
                    TileExecutionRecord(
                        round_index=round_plan['round_index'],
                        tile_index=tile_index,
                        region_global=region,
                        known_fraction=float((1.0 - known_mask).mean().detach().cpu()),
                    )
                )
            )
        generated_canvas = torch.where(
            round_weights.squeeze(0).squeeze(0).unsqueeze(0).unsqueeze(0) > 0,
            round_accum / torch.clamp(round_weights.squeeze(0).squeeze(0).unsqueeze(0).unsqueeze(0), min=1e-6),
            working_canvas,
        )
        working_generate_mask = working_known[0, :, 0].unsqueeze(1)
        working_canvas = working_canvas * (1.0 - working_generate_mask) + generated_canvas * working_generate_mask
        round_region = round_plan['canvas_region_global']
        working_known[
            :,
            :,
            :,
            round_region['top']:round_region['top'] + round_region['height'],
            round_region['left']:round_region['left'] + round_region['width'],
        ] = 0.0

    sample_video_path = _write_video_tensor(working_canvas.detach().cpu(), samples_dir / 'outpaint.mp4', fps=config.fps)
    metadata = {
        'run_name': run_name,
        'generation_mode': 'outpaint',
        'model': loaded.summary() if hasattr(loaded, 'summary') else {'pipeline_class': type(pipe).__name__},
        'prompt': config.prompt,
        'negative_prompt': config.negative_prompt,
        'height': config.canvas_height,
        'width': config.canvas_width,
        'num_frames': normalized_num_frames,
        'num_inference_steps': config.num_inference_steps,
        'guidance_scale': config.guidance_scale,
        'fps': config.fps,
        'seed': config.seed,
        'plan_path': str(plan_path),
        'tile_records': tile_records,
        'elapsed_sec': round(perf_counter() - started, 3),
        'source_video_archive_path': str(source_video_archive_path) if source_video_archive_path else None,
        'sample_video_path': str(sample_video_path),
    }
    metadata_path = write_json_report(logs_dir / 'metadata.json', metadata)
    return OutpaintRunArtifacts(
        run_root=str(run_root),
        original_dir=str(original_dir),
        samples_dir=str(samples_dir),
        logs_dir=str(logs_dir),
        prompt_path=str(prompt_path),
        negative_prompt_path=str(negative_prompt_path),
        metadata_path=str(metadata_path),
        sample_video_path=str(sample_video_path),
        source_video_archive_path=str(source_video_archive_path) if source_video_archive_path else None,
        plan_path=str(plan_path),
    )
