# Wan FYC replacement / removal matrix

| Status | Active replacement | Removed legacy path |
| --- | --- | --- |
| replaced + removed | `wancanvas/inference/outpaint_runner.py` | `wancanvas/inference/ti2v_runner.py` |
| replaced + removed | `wancanvas/train/smoke_trainer.py` | `wancanvas/train/dry_run.py` |
| replaced + removed | runtime smoke/inference tests under `WanCanvas/tests/` | `tests/test_ti2v_runner.py` |
| replaced + removed | smoke trainer tests under `WanCanvas/tests/` | `tests/test_training_dry_run.py` |
| replaced + removed | `scripts/run_wan_outpaint.py` | `scripts/run_ti2v_inference.py` |
| replaced + removed | `scripts/train_wan_outpaint.py` | `scripts/train_wancanvas.py` |
| replaced + removed | `configs/infer-outpaint-real.yaml` | `configs/infer-real-ti2v.yaml` |
| replaced + removed | `configs/infer-outpaint-smoke.yaml` | `configs/infer-smoke.yaml` |
| replaced + removed | `configs/train-outpaint-smoke.yaml` | `configs/train-skeleton.yaml` |
| replaced + removed | `wancanvas/train/__init__.py` + smoke runtime | `wancanvas/train/contracts.py` |
