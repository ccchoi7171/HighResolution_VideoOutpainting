# Wan + FYC architecture note

## Chosen execution path
- **Text path:** prompt text is encoded by the active Wan runtime.
- **FYC path:** layout / geometry / mask tokens are concatenated and projected into Wan's `encoder_hidden_states_image` stream.
- **Known-region path:** the denoiser input carries noisy latents plus a known-region package (`mask + condition latents`) so preserved content remains explicit at every step.

## Phase-1 trainable modules
- layout encoder
- geometry encoder
- mask summary encoder
- FYC token projector inside `WanOutpaintWrapper`

## Frozen modules in phase 1
- pretrained Wan backbone when a real runtime is available
- the smoke runtime transformer is treated as execution infrastructure, not the main optimization target

## Runtime modes
- `pretrained`: use local/cached Wan2.2 weights through `WanImageToVideoPipeline`
- `smoke`: use the built-in tiny Wan transformer for offline verification
- `auto`: prefer local pretrained weights, otherwise fall back to smoke

## High-resolution strategy
1. build the multi-round canvas plan
2. generate per-tile updates
3. merge with gaussian overlap weights
4. re-apply known-region preservation after each denoise step and round
