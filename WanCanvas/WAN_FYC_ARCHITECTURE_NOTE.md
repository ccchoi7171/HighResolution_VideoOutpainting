# WanCanvas FYC -> Wan Architecture Note

## Real conditioning path

The active tree no longer stops at a bridge-only proof surface.

### FYC token path
- `layout_encoder`
- `geometry_encoder`
- `mask_summary_encoder`

These produce FYC condition tokens which are projected into Wan's **real `encoder_hidden_states_image`** input.

### Known-region latent path
- `condition_video` is encoded into Wan latents
- `known_mask` is resized to a latent-space mask
- execution uses:

```text
latent_model_input = (1 - latent_mask) * condition_latents + latent_mask * noisy_latents
```

### Train path
- build `FYCOutpaintSample`
- bridge to runtime tensors
- encode target / condition latents
- run Wan forward
- compute diffusion + known-region + seam losses
- `backward()` + `optimizer.step()`

### Inference path
- read anchor/source video
- plan multi-round canvas expansion
- execute tile-wise outpainting
- gaussian overlap merge
- write final outpainted video

## Replacement / removal matrix

The following files were replaced and then removed from the repo so the
active tree only reflects the current WanCanvas execution surface:

| Old path | Replacement |
| --- | --- |
| `wancanvas/train/contracts.py` | `wancanvas/train/smoke_trainer.py` |
| `wancanvas/train/dry_run.py` | `wancanvas/train/smoke_trainer.py` |
| `wancanvas/inference/ti2v_runner.py` | `wancanvas/inference/outpaint_runner.py` |
| `scripts/run_ti2v_inference.py` | `scripts/run_wan_outpaint.py` |
| `scripts/train_wancanvas.py` | `scripts/train_wan_outpaint.py` |
| `configs/infer-real-ti2v.yaml` | `configs/infer-outpaint-real.yaml` |
| `configs/infer-smoke.yaml` | `configs/infer-outpaint-smoke.yaml` |
| `configs/train-skeleton.yaml` | `configs/train-outpaint-smoke.yaml` |

## Runtime assumptions
- conda env: `wancanvas`
- model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- official runtime source: Diffusers Wan implementation
