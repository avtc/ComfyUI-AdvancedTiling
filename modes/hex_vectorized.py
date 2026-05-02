"""Vectorized hex tiling convenience wrappers for full-grid usage."""

import numpy as np
from .hex import hex_remap_batch


def hex_tiling_vectorized(
    width: int,
    height: int,
    rotation: float,
    hex_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized hexagonal tiling for all pixels at once.

    Convenience wrapper around hex_remap_batch for full-grid usage.
    Produces results identical to the scalar hex_tiling() per-pixel function.

    :param width: Grid width
    :param height: Grid height
    :param rotation: Rotation angle in degrees
    :param hex_size: Hex radius at this resolution
    :return: (mapped_x, mapped_y) as int64 arrays of shape (height, width)
    """
    cx = np.arange(width, dtype=np.float64) - width // 2
    cy = np.arange(height, dtype=np.float64) - height // 2
    grid_x, grid_y = np.meshgrid(cx, cy, indexing='xy')
    new_x, new_y = hex_remap_batch(grid_x, grid_y, hex_size, width, height, rotation)
    return new_x, new_y
