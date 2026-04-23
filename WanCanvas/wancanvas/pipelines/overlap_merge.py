from __future__ import annotations

from math import exp, pi, sqrt


def gaussian_weights_2d(tile_width: int, tile_height: int, *, variance: float = 0.01) -> list[list[float]]:
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile dimensions must be positive")
    midpoint_x = (tile_width - 1) / 2
    midpoint_y = (tile_height - 1) / 2

    def _pdf(delta: float, scale: float) -> float:
        return exp(-(delta * delta) / (scale * scale) / (2 * variance)) / sqrt(2 * pi * variance)

    x_probs = [_pdf(x - midpoint_x, tile_width) for x in range(tile_width)]
    y_probs = [_pdf(y - midpoint_y, tile_height) for y in range(tile_height)]
    return [[y_probs[y] * x_probs[x] for x in range(tile_width)] for y in range(tile_height)]
