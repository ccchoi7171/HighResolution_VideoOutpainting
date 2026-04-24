from __future__ import annotations

import argparse
from dataclasses import asdict

from wancanvas.backbones.wan_loader import WanLoader
from wancanvas.data.outpaint_dataset import DatasetRecord, WanCanvasDataset
from wancanvas.data.samplers import AnchorTargetSamplingConfig
from wancanvas.models.wan_outpaint_wrapper import WanOutpaintWrapper
from wancanvas.train import SmokeTrainer
from wancanvas.utils.logging import dump_json

try:
    import torch
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    torch = None
    imageio = None


def load_video(record: DatasetRecord):
    if torch is None or imageio is None:
        raise RuntimeError('torch and imageio are required for train_wan_outpaint.py')
    reader = imageio.get_reader(record.payload['video_path'])
    frames = []
    try:
        for frame in reader:
            frames.append(torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0)
            if len(frames) >= record.frame_count:
                break
    finally:
        reader.close()
    while len(frames) < record.frame_count:
        frames.append(frames[-1].clone())
    return torch.stack(frames, dim=0)


def crop_video(video: torch.Tensor, region):
    return video[:, :, region.top:region.bottom, region.left:region.right]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--frame-height", type=int, required=True)
    parser.add_argument("--frame-width", type=int, required=True)
    parser.add_argument("--frame-count", type=int, default=17)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--target-height", type=int, default=512)
    parser.add_argument("--target-width", type=int, default=512)
    parser.add_argument("--anchor-height", type=int, default=384)
    parser.add_argument("--anchor-width", type=int, default=384)
    parser.add_argument("--runtime-variant", choices=["auto", "pretrained", "smoke"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    args = parser.parse_args()

    dataset = WanCanvasDataset(
        records=[
            DatasetRecord(
                source_id="train-demo",
                prompt=args.prompt,
                frame_height=args.frame_height,
                frame_width=args.frame_width,
                frame_count=args.frame_count,
                fps=args.fps,
                payload={"video_path": args.video_path},
            )
        ],
        sampling_config=AnchorTargetSamplingConfig(
            target_size=(args.target_height, args.target_width),
            anchor_size=(args.anchor_height, args.anchor_width),
            seed=11,
        ),
        frame_loader=load_video,
        cropper=crop_video,
    )
    sample = dataset[0]
    wrapper = WanOutpaintWrapper(WanLoader())
    trainer = SmokeTrainer(wrapper=wrapper)
    runtime = wrapper.load_runtime(runtime_variant=args.runtime_variant, device=args.device, torch_dtype=args.torch_dtype)
    print(dump_json(asdict(trainer.run_once(sample, runtime=runtime))))
