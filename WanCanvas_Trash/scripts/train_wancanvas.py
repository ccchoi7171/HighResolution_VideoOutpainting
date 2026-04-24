from __future__ import annotations

from dataclasses import asdict

from wancanvas.backbones.wan_loader import WanLoader
from wancanvas.data.outpaint_dataset import DatasetRecord, WanCanvasDataset
from wancanvas.data.samplers import AnchorTargetSamplingConfig
from wancanvas.models.wan_outpaint_wrapper import WanOutpaintWrapper
from wancanvas.train import DryRunTrainer, TrainStepBuilder
from wancanvas.utils.logging import dump_json


if __name__ == "__main__":
    dataset = WanCanvasDataset(
        records=[DatasetRecord(source_id="train-demo", prompt="expand the frame", frame_height=720, frame_width=1280)],
        sampling_config=AnchorTargetSamplingConfig(target_size=(512, 512), anchor_size=(384, 384), seed=11),
    )
    sample = dataset[0]
    trainer = DryRunTrainer(
        wrapper=WanOutpaintWrapper(WanLoader()),
        step_builder=TrainStepBuilder(("layout_encoder", "geometry_encoder", "condition_adapter", "wan_outpaint_wrapper")),
    )
    print(dump_json(asdict(trainer.run_once(sample))))
