# WanCanvas: Wan-grounded FYC video outpainting

WanCanvas is the cleaned execution surface for adapting **Follow-Your-Canvas (FYC)** conditioning into a **Wan / DiT** video-outpainting path.

## What is now true
- `WanOutpaintWrapper.forward()` performs a real Wan-transformer denoiser call.
- FYC layout / geometry / mask tokens are consumed through Wan's `encoder_hidden_states_image` path.
- Inference uses anchor video, known-region masks, multi-round planning, tile execution, and gaussian overlap merge.
- Training smoke performs `forward + backward + optimizer.step()`.
- Superseded TI2V-only / dry-run-only surfaces were moved under `WanCanvas_Trash/`.

## Runtime modes
- `auto`: prefer cached Wan2.2 weights, otherwise fall back to the built-in smoke runtime
- `pretrained`: force the local/cached Wan2.2 runtime
- `smoke`: use the tiny offline Wan transformer for fast verification

The smoke runtime exists so the repo can be verified without downloading the full 5B weights. The execution path stays structurally aligned with Wan: text conditioning, image-conditioning tokens, scheduler-driven denoising, and latent/video conversion.

## Active tree
```text
WanCanvas/
├── configs/
├── docs/
├── scripts/
├── tests/
└── wancanvas/
    ├── backbones/
    ├── data/
    ├── inference/
    ├── models/
    ├── pipelines/
    ├── train/
    └── utils/
```

## Main components
- `wancanvas/models/wan_outpaint_wrapper.py` — Wan forward path, FYC token projection, latent conditioning, generation loop
- `wancanvas/models/fyc_sample_bridge.py` — sample -> Wan request bridge
- `wancanvas/inference/outpaint_runner.py` — real outpaint inference runner
- `wancanvas/train/smoke_trainer.py` — executable smoke training path
- `wancanvas/pipelines/wan_outpaint_pipeline.py` — multi-round planning and dry-run invariants

## Docs
- Architecture note: `docs/wan-fyc-architecture-note.md`
- Trash matrix: `docs/wan-fyc-trash-matrix.md`

## Verification
Current regression coverage:
- `python -m compileall WanCanvas/wancanvas WanCanvas/scripts WanCanvas/tests`
- `PYTHONPATH=WanCanvas conda run --no-capture-output -n wancanvas python -m unittest discover -s WanCanvas/tests -p 'test_*.py'`

The current suite passes with **32 tests**.

## Quick start
### Smoke inference
```bash
PYTHONPATH=WanCanvas conda run --no-capture-output -n wancanvas \
  python WanCanvas/scripts/run_wan_outpaint.py \
  --prompt "expand the frame" \
  --source-video path/to/source.mp4 \
  --runtime-variant smoke
```

### Smoke training
```bash
PYTHONPATH=WanCanvas conda run --no-capture-output -n wancanvas \
  python WanCanvas/scripts/train_wan_outpaint.py \
  --video-path path/to/source.mp4 \
  --prompt "expand the frame" \
  --frame-height 720 \
  --frame-width 1280 \
  --runtime-variant smoke
```

### Cached pretrained runtime
```bash
PYTHONPATH=WanCanvas conda run --no-capture-output -n wancanvas \
  python WanCanvas/scripts/run_wan_outpaint.py \
  --prompt "cinematic snowy forest panorama" \
  --source-video path/to/source.mp4 \
  --runtime-variant pretrained
```
