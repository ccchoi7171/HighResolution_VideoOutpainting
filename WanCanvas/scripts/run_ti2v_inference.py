from __future__ import annotations

import argparse

from wancanvas.inference import Ti2VInferenceConfig, run_ti2v_inference
from wancanvas.inference.ti2v_runner import DEFAULT_NEGATIVE_PROMPT
from wancanvas.utils.logging import dump_json


DEFAULT_PROMPT = (
    "A majestic polar bear slowly walks through a snowy forest, cinematic tracking shot, "
    "soft winter sunlight, floating snow particles, realistic fur, calm atmosphere"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--output-root", default="WanCanvas/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--source-image", default="FollowYourCanvas/infer/temp.jpg")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--pipeline-class", default="WanPipeline")
    parser.add_argument("--cache-dir", default="WanCanvas/.hf-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--scheduler-name", default="FlowMatchEulerDiscreteScheduler")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    args = parser.parse_args()

    config = Ti2VInferenceConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        output_root=args.output_root,
        run_name=args.run_name,
        source_image=args.source_image,
        model_id=args.model_id,
        pipeline_class=args.pipeline_class,
        cache_dir=args.cache_dir,
        device=args.device,
        torch_dtype=args.torch_dtype,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fps=args.fps,
        seed=args.seed,
        flow_shift=args.flow_shift,
        scheduler_name=args.scheduler_name,
        local_files_only=args.local_files_only,
        enable_model_cpu_offload=args.enable_model_cpu_offload,
    )
    artifacts = run_ti2v_inference(config)
    print(dump_json(artifacts.to_dict()))
