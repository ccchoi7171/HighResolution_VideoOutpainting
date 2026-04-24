from __future__ import annotations


def alignment_quantum(vae_scale_factor_spatial: int, patch_size: int) -> int:
    if vae_scale_factor_spatial <= 0 or patch_size <= 0:
        raise ValueError("alignment factors must be positive")
    return vae_scale_factor_spatial * patch_size


def estimate_latent_hw(height: int, width: int, *, vae_scale_factor_spatial: int = 8) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    return height // vae_scale_factor_spatial, width // vae_scale_factor_spatial


def estimate_latent_frames(frame_count: int, *, vae_scale_factor_temporal: int = 4) -> int:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if vae_scale_factor_temporal <= 0:
        raise ValueError("vae_scale_factor_temporal must be positive")
    return ((frame_count - 1) // vae_scale_factor_temporal) + 1
