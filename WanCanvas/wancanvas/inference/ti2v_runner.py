from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import shutil

import numpy as np
from PIL import Image
from diffusers.utils import export_to_video

from ..backbones.wan_loader import WanLoader
from ..data.contracts import Rect
from ..models.wan_outpaint_wrapper import WanOutpaintWrapper
from ..pipelines.size_alignment import SizeAlignmentRule, validate_spatial_size
from ..pipelines.wan_outpaint_pipeline import OutpaintRequest, WanOutpaintPipeline
from ..utils.logging import write_json_report

DEFAULT_NEGATIVE_PROMPT = (
    'overexposed, static shot, blurry details, subtitles, watermark, worst quality, low quality, '
    'jpeg artifacts, malformed anatomy, extra limbs, fused fingers, messy background'
)


@dataclass(slots=True)
class Ti2VInferenceConfig:
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    output_root: str = 'runs/wancanvas'
    run_name: str | None = None
    source_image: str | None = None
    model_id: str = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
    pipeline_class: str = 'WanPipeline'
    cache_dir: str = '.cache/wancanvas'
    device: str = 'cuda'
    torch_dtype: str = 'bfloat16'
    height: int = 704
    width: int = 1280
    num_frames: int = 49
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    fps: int = 16
    seed: int = 7
    flow_shift: float = 5.0
    scheduler_name: str | None = 'FlowMatchEulerDiscreteScheduler'
    create_reference_hold_video: bool = True
    hold_video_frames: int = 24
    local_files_only: bool = False
    enable_model_cpu_offload: bool = False

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError('prompt must not be empty')
        if self.height <= 0 or self.width <= 0:
            raise ValueError('height and width must be positive')
        if self.num_frames <= 0 or self.num_inference_steps <= 0:
            raise ValueError('num_frames and num_inference_steps must be positive')
        if self.fps <= 0:
            raise ValueError('fps must be positive')


@dataclass(slots=True)
class Ti2VRunArtifacts:
    run_root: str
    original_dir: str
    samples_dir: str
    logs_dir: str
    prompt_path: str
    negative_prompt_path: str
    metadata_path: str
    sample_video_path: str
    reference_image_path: str | None = None
    reference_hold_video_path: str | None = None
    plan_path: str | None = None
    generation_mode: str = 't2v'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slugify(text: str, max_len: int = 48) -> str:
    safe: list[str] = []
    for char in text.lower():
        if char.isalnum():
            safe.append(char)
        elif char in {' ', '-', '_'}:
            safe.append('-')
    slug = ''.join(safe).strip('-') or 'wan-run'
    while '--' in slug:
        slug = slug.replace('--', '-')
    return slug[:max_len].rstrip('-') or 'wan-run'


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


def _resolve_run_name(config: Ti2VInferenceConfig) -> str:
    if config.run_name:
        return config.run_name
    return f'ti2v-{_slugify(config.prompt)}-s{config.seed}'


def _ensure_output_dirs(output_root: str, run_name: str) -> tuple[Path, Path, Path, Path]:
    run_root = Path(output_root) / run_name
    original_dir = run_root / 'original'
    samples_dir = run_root / 'samples'
    logs_dir = run_root / 'logs'
    for path in (original_dir, samples_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return run_root, original_dir, samples_dir, logs_dir


def _copy_reference_image(source_image: str | None, original_dir: Path) -> Path | None:
    if not source_image:
        return None
    src = Path(source_image)
    if not src.exists():
        raise FileNotFoundError(f'reference image not found: {src}')
    destination = original_dir / f'reference{src.suffix or ".png"}'
    shutil.copy2(src, destination)
    return destination


def _write_reference_hold_video(reference_image_path: Path | None, output_path: Path, *, fps: int, frames: int) -> Path | None:
    if reference_image_path is None:
        return None
    import imageio.v2 as imageio

    frame = np.array(Image.open(reference_image_path).convert('RGB'))
    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for _ in range(frames):
            writer.append_data(frame)
    finally:
        writer.close()
    return output_path


def _build_planning_artifact(config: Ti2VInferenceConfig, logs_dir: Path) -> Path:
    wrapper = WanOutpaintWrapper(WanLoader())
    pipeline = WanOutpaintPipeline(wrapper=wrapper, size_rule=SizeAlignmentRule())
    anchor = Rect(top=config.height // 4, left=config.width // 4, height=config.height // 2, width=config.width // 2)
    plan = pipeline.plan_request(
        OutpaintRequest(
            prompt=config.prompt,
            frame_count=_normalize_num_frames(config.num_frames),
            fps=config.fps,
            canvas_height=config.height,
            canvas_width=config.width,
            anchor_region=anchor,
            tile_height=min(720, config.height),
            tile_width=min(720, config.width),
            overlap_height=min(176, max(config.height // 8, 0)),
            overlap_width=min(176, max(config.width // 8, 0)),
            extras={
                'generation_mode': 'planning-only',
                'runtime_pipeline_class': config.pipeline_class,
            },
        )
    )
    return write_json_report(logs_dir / 'wancanvas_plan.json', plan)


def _build_runtime_metadata(
    *,
    config: Ti2VInferenceConfig,
    loaded_summary: dict[str, Any],
    run_name: str,
    normalized_num_frames: int,
    prompt_path: Path,
    negative_prompt_path: Path,
    sample_video_path: Path,
    plan_path: Path,
    reference_image_path: Path | None,
    reference_hold_video_path: Path | None,
    elapsed_sec: float,
    peak_memory_gb: float | None,
) -> dict[str, Any]:
    notes = [
        'The retained inference surface is intentionally minimal: prompt/reference capture, planning artifact generation, and Wan TI2V runtime execution.',
        'If a source image is provided, WanCanvas archives it for traceability; the current verified runtime path remains the official Wan TI2V generation surface.',
    ]
    if normalized_num_frames != config.num_frames:
        notes.append(
            f'num_frames normalized from {config.num_frames} to {normalized_num_frames} to satisfy the model frame-count constraint.'
        )
    return {
        'run_name': run_name,
        'generation_mode': 't2v',
        'model': loaded_summary,
        'prompt': config.prompt,
        'negative_prompt': config.negative_prompt,
        'prompt_path': str(prompt_path),
        'negative_prompt_path': str(negative_prompt_path),
        'source_image': str(reference_image_path) if reference_image_path else None,
        'reference_hold_video': str(reference_hold_video_path) if reference_hold_video_path else None,
        'height': config.height,
        'width': config.width,
        'num_frames': normalized_num_frames,
        'requested_num_frames': config.num_frames,
        'num_inference_steps': config.num_inference_steps,
        'guidance_scale': config.guidance_scale,
        'fps': config.fps,
        'seed': config.seed,
        'sample_video_path': str(sample_video_path),
        'plan_path': str(plan_path),
        'elapsed_sec': round(elapsed_sec, 3),
        'peak_memory_gb': peak_memory_gb,
        'notes': notes,
    }


def run_ti2v_inference(config: Ti2VInferenceConfig) -> Ti2VRunArtifacts:
    config.validate()
    rule = SizeAlignmentRule()
    aligned, errors = validate_spatial_size(config.height, config.width, rule)
    if not aligned:
        raise ValueError('size alignment failed: ' + '; '.join(errors))

    normalized_num_frames = _normalize_num_frames(config.num_frames)
    run_name = _resolve_run_name(config)
    run_root, original_dir, samples_dir, logs_dir = _ensure_output_dirs(config.output_root, run_name)

    prompt_path = original_dir / 'prompt.txt'
    prompt_path.write_text(config.prompt + '\n', encoding='utf-8')
    negative_prompt_path = original_dir / 'negative_prompt.txt'
    negative_prompt_path.write_text(config.negative_prompt + '\n', encoding='utf-8')

    reference_image_path = _copy_reference_image(config.source_image, original_dir)
    reference_hold_video_path = None
    if config.create_reference_hold_video and reference_image_path is not None:
        reference_hold_video_path = _write_reference_hold_video(
            reference_image_path,
            original_dir / 'reference_hold.mp4',
            fps=config.fps,
            frames=config.hold_video_frames,
        )

    plan_path = _build_planning_artifact(config, logs_dir)

    import torch

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    loader = WanLoader()
    started = perf_counter()
    loaded = loader.load_pipeline(
        model_id=config.model_id,
        pipeline_class_name=config.pipeline_class,
        device=config.device,
        torch_dtype=config.torch_dtype,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
        flow_shift=config.flow_shift,
        scheduler_name=config.scheduler_name,
        enable_model_cpu_offload=config.enable_model_cpu_offload,
    )
    pipe = loaded.pipeline
    try:
        generator = torch.Generator(device=config.device).manual_seed(config.seed)
    except Exception:
        generator = torch.Generator().manual_seed(config.seed)

    output = pipe(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        height=config.height,
        width=config.width,
        num_frames=normalized_num_frames,
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
        generator=generator,
    ).frames[0]

    sample_video_path = samples_dir / 'generated.mp4'
    export_to_video(output, str(sample_video_path), fps=config.fps)
    elapsed_sec = perf_counter() - started

    peak_memory_gb = None
    if torch.cuda.is_available():
        peak_memory_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        torch.cuda.empty_cache()

    metadata = _build_runtime_metadata(
        config=config,
        loaded_summary={
            **loaded.summary(),
            'scheduler_name': config.scheduler_name or loaded.pipeline.scheduler.__class__.__name__,
            'alignment_quantum': rule.quantum,
        },
        run_name=run_name,
        normalized_num_frames=normalized_num_frames,
        prompt_path=prompt_path,
        negative_prompt_path=negative_prompt_path,
        sample_video_path=sample_video_path,
        plan_path=plan_path,
        reference_image_path=reference_image_path,
        reference_hold_video_path=reference_hold_video_path,
        elapsed_sec=elapsed_sec,
        peak_memory_gb=peak_memory_gb,
    )
    metadata_path = write_json_report(logs_dir / 'run_metadata.json', metadata)

    return Ti2VRunArtifacts(
        run_root=str(run_root),
        original_dir=str(original_dir),
        samples_dir=str(samples_dir),
        logs_dir=str(logs_dir),
        prompt_path=str(prompt_path),
        negative_prompt_path=str(negative_prompt_path),
        metadata_path=str(metadata_path),
        sample_video_path=str(sample_video_path),
        reference_image_path=str(reference_image_path) if reference_image_path else None,
        reference_hold_video_path=str(reference_hold_video_path) if reference_hold_video_path else None,
        plan_path=str(plan_path),
        generation_mode='t2v',
    )
