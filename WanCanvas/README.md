# WanCanvas: Wan-based DiT Scaffold for Video Outpainting

WanCanvas is the final cleaned code release for our Wan-based DiT reinterpretation of the core **Follow-Your-Canvas (FYC)** ideas inside this repository.

This subtree is intentionally kept small and paper-ready. It retains only the surfaces that are still necessary to understand, inspect, and verify the current migration from an FYC-style outpainting formulation to a **Wan2.2 TI2V-5B Diffusers** backbone.

## Overview

The retained codebase focuses on four concrete goals:

1. **FYC-style conditioning preservation**
   - layout conditioning
   - relative-geometry conditioning
   - known-region / mask-summary conditioning

2. **Wan runtime adaptation**
   - Wan runtime inspection
   - Wan pipeline loading
   - Wan wrapper input preparation

3. **Planning and dry-run validation**
   - outpaint canvas/tile planning
   - non-mutating dry-run execution
   - pre-training-entry contract validation

4. **Minimal runnable inference surface**
   - a verified TI2V runtime entrypoint for the Wan backbone
   - prompt/reference archiving and run metadata capture

## Release Scope

This release is meant to support **code inspection, supplementary material, and reproducibility of the current migration stage**.

The retained `WanCanvas/` tree includes:
- Wan runtime loading and validation
- FYC-style conditioning encoders and bridge logic
- outpaint planning utilities
- a minimal TI2V inference runner
- a merged, non-mutating training-entry dry-run scaffold
- a compact regression test suite for the kept surfaces

The release intentionally excludes older proof-manifest layers, smoke-only scripts, generated artifacts, and superseded contract ladders so that the final subtree stays concise and reviewable.

## Current Technical Status

### What is implemented
- FYC-style conditioning has been mapped into a Wan-based DiT-oriented scaffold.
- The repository contains a cleaned planning / bridge / wrapper / dry-run stack.
- The retained TI2V runner executes the verified Wan runtime path.
- The training side exposes a **non-mutating pre-training boundary** with explicit checks for the next execution stage.

### What is **not** claimed by this release
- This branch is **not** a full end-to-end training release.
- The public training path is still a **dry-run / pre-training-entry scaffold**.
- The next step exposed by the retained training surface is still:
  - `real_forward_graph_materialization`
- The current release does **not** claim completed execution of:
  - real forward graph materialization for training
  - `loss.backward()`
  - `optimizer.step()`
  - final trained-model quality comparisons against the original FYC baseline
- The retained TI2V runner archives an optional source image for traceability, but the current released runtime path should be read as a **minimal Wan TI2V execution surface**, not as a claim of full source-image-conditioned FYC parity.

## Repository Layout

```text
WanCanvas/
├── configs/                # minimal retained config surface
├── scripts/                # runtime inspection, dry-run, TI2V entrypoints
├── tests/                  # compact regression suite for retained code
└── wancanvas/
    ├── backbones/          # Wan runtime inspection and loading
    ├── data/               # sample/data contracts and geometry helpers
    ├── inference/          # minimal TI2V runtime entrypoint
    ├── models/             # FYC conditioning and Wan wrapper adaptation
    ├── pipelines/          # planning, known-region handling, scheduling
    ├── train/              # merged dry-run training-entry surface
    └── utils/              # lightweight shared helpers
```

## Main Components

### `wancanvas/backbones/`
Runtime inspection and loading utilities for the Wan Diffusers stack.

### `wancanvas/models/`
Core conditioning and adaptation logic:
- layout encoding
- geometry encoding
- mask-summary encoding
- FYC sample bridge
- Wan wrapper integration

### `wancanvas/pipelines/`
Planning and execution scaffolding for outpainting:
- size alignment
- known-region preservation logic
- window scheduling
- outpaint request planning

### `wancanvas/train/contracts.py`
Merged training contract surface that replaces the earlier fragmented micro-contract ladder.

### `wancanvas/train/dry_run.py`
Merged dry-run trainer surface for the current non-mutating pre-training-entry stage.

### `wancanvas/inference/ti2v_runner.py`
Minimal runnable TI2V entrypoint for the retained Wan runtime path.

## Verification Status

The cleaned release was validated with the following checks:

- `python -m compileall WanCanvas/wancanvas WanCanvas/scripts WanCanvas/tests`
- `PYTHONPATH=WanCanvas conda run --no-capture-output -n wancanvas python -m unittest discover -s WanCanvas/tests -p 'test_*.py'`
- `PYTHONPATH=WanCanvas python WanCanvas/scripts/train_wancanvas.py`
- `PYTHONPATH=WanCanvas python WanCanvas/scripts/inspect_wan_runtime.py`

At the time of final cleanup, the retained regression suite passed with **32/32 tests**.

## Runtime Defaults

To keep the source tree clean, the retained inference runner defaults to:
- outputs: `runs/wancanvas/`
- cache: `.cache/wancanvas/`

## Quick Start

### 1. Environment

Use the provided environment description and runtime requirements as the starting point for the `wancanvas` environment.

### 2. Compile check

```bash
python -m compileall WanCanvas/wancanvas WanCanvas/scripts WanCanvas/tests
```

### 3. Run the retained regression suite

```bash
PYTHONPATH=WanCanvas conda run --no-capture-output -n wancanvas \
  python -m unittest discover -s WanCanvas/tests -p 'test_*.py'
```

### 4. Inspect the Wan runtime

```bash
PYTHONPATH=WanCanvas python WanCanvas/scripts/inspect_wan_runtime.py
```

### 5. Run the training dry-run scaffold

```bash
PYTHONPATH=WanCanvas python WanCanvas/scripts/train_wancanvas.py
```

### 6. Run TI2V inference

```bash
PYTHONPATH=WanCanvas python WanCanvas/scripts/run_ti2v_inference.py \
  --prompt "cinematic snowy forest panorama"
```

## Notes for Readers and Reviewers

- This subtree should be treated as the **final cleaned code surface**, not as a dump of every intermediate artifact produced during migration.
- Older archive/proof/smoke material was intentionally removed from the final source tree to keep the release readable.
- The release is strongest as a record of the **architecture migration and validation boundary**, rather than as a claim of a fully completed training pipeline.

## Citation

If you use this code in academic work, please cite the accompanying paper or project release associated with this repository.
