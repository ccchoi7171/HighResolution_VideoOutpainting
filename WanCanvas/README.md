# WanCanvas

WanCanvas is the cleaned Wan-based DiT outpainting scaffold extracted from the FYC-to-Wan migration work.

## Scope

`WanCanvas/` now keeps only the code paths that are still meaningful for:
- Wan runtime loading
- FYC-style conditioning and sample bridging
- outpaint planning / dry-run execution
- non-mutating training-entry validation
- minimal TI2V inference execution

Archived cleanup byproducts were moved out of the final source tree so `WanCanvas/` stays paper- and git-ready.

## Layout

```text
WanCanvas/
├── configs/
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

## Kept runtime surfaces

### Source
- `wancanvas/backbones/` — runtime inspection + Wan loader
- `wancanvas/data/` — dataset/sample contracts and geometry helpers
- `wancanvas/models/` — FYC conditioning bridge and Wan wrapper
- `wancanvas/pipelines/` — known-region handling, size alignment, scheduler, outpaint planning
- `wancanvas/train/contracts.py` — merged dry-run training contract surface
- `wancanvas/train/dry_run.py` — merged dry-run trainer surface
- `wancanvas/inference/ti2v_runner.py` — minimal TI2V runtime entrypoint

### Scripts
- `scripts/inspect_wan_runtime.py`
- `scripts/train_wancanvas.py`
- `scripts/run_ti2v_inference.py`

### Configs
- `configs/base.yaml`
- `configs/infer-real-ti2v.yaml`
- `configs/infer-smoke.yaml`
- `configs/model-ti2v-5b.yaml`
- `configs/train-skeleton.yaml`

## Default runtime output locations

To keep the source tree clean, the retained inference runner defaults to:
- outputs: `runs/wancanvas/`
- cache: `.cache/wancanvas/`

## Quick checks

### Unit tests

```bash
PYTHONPATH=WanCanvas conda run --no-capture-output -n wancanvas \
  python -m unittest discover -s WanCanvas/tests -p 'test_*.py'
```

### Compile check

```bash
python -m compileall WanCanvas/wancanvas WanCanvas/scripts WanCanvas/tests
```

### Training dry run

```bash
PYTHONPATH=WanCanvas python WanCanvas/scripts/train_wancanvas.py
```

### TI2V inference

```bash
PYTHONPATH=WanCanvas python WanCanvas/scripts/run_ti2v_inference.py \
  --prompt "cinematic snowy forest panorama"
```

## Notes

- The retained training path is still a **non-mutating dry-run / pre-training-entry scaffold**.
- The retained TI2V runner is intentionally minimal and honest: it archives prompt/reference assets, writes a planning artifact, and executes the verified Wan TI2V runtime path.
- Older proof manifests, smoke scripts, and superseded contract ladders were intentionally excluded from the final source tree.
