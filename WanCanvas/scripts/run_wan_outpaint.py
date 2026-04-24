from __future__ import annotations

import argparse

from wancanvas.inference import OutpaintInferenceConfig, run_outpaint_inference
from wancanvas.utils.logging import dump_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--output-root", default="WanCanvas/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--pipeline-class", default="WanImageToVideoPipeline")
    parser.add_argument("--cache-dir", default="WanCanvas/.hf-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--canvas-height", type=int, default=704)
    parser.add_argument("--canvas-width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--scheduler-name", default="FlowMatchEulerDiscreteScheduler")
    parser.add_argument("--tile-height", type=int, default=720)
    parser.add_argument("--tile-width", type=int, default=720)
    parser.add_argument("--overlap-height", type=int, default=176)
    parser.add_argument("--overlap-width", type=int, default=176)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--anchor-top", type=int, default=None)
    parser.add_argument("--anchor-left", type=int, default=None)
    parser.add_argument("--anchor-height", type=int, default=None)
    parser.add_argument("--anchor-width", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--runtime-variant", choices=["auto", "pretrained", "smoke"], default="auto")
    args = parser.parse_args()

    config = OutpaintInferenceConfig(
        prompt=args.prompt,
        source_video=args.source_video,
        negative_prompt=args.negative_prompt or "",
        output_root=args.output_root,
        run_name=args.run_name,
        model_id=args.model_id,
        pipeline_class=args.pipeline_class,
        cache_dir=args.cache_dir,
        device=args.device,
        torch_dtype=args.torch_dtype,
        canvas_height=args.canvas_height,
        canvas_width=args.canvas_width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fps=args.fps,
        seed=args.seed,
        flow_shift=args.flow_shift,
        scheduler_name=args.scheduler_name,
        tile_height=args.tile_height,
        tile_width=args.tile_width,
        overlap_height=args.overlap_height,
        overlap_width=args.overlap_width,
        rounds=args.rounds,
        anchor_top=args.anchor_top,
        anchor_left=args.anchor_left,
        anchor_height=args.anchor_height,
        anchor_width=args.anchor_width,
        local_files_only=args.local_files_only,
        enable_model_cpu_offload=args.enable_model_cpu_offload,
        runtime_variant=args.runtime_variant,
    )
    artifacts = run_outpaint_inference(config)
    print(dump_json(artifacts.to_dict()))
