"""
Rectangular (toroidal) tiling implementation

Wraps coordinates modularly: right edge wraps to left, bottom wraps to top.
"""

import math
from . import Settings


def compute_float_rect_dims(
    W: int,
    H: int,
    settings: Settings,
    vae_factor: int,
) -> tuple[float, float]:
    """Compute float-point rectangle dimensions from scale, min_margin, and divisible_by.

    Pipeline: scale * dim → subtract 2 * min_margin → round down for divisible_by.
    Returns float (work_w, work_h).
    """
    work_w = W * settings.scale - 2 * settings.min_margin
    work_h = H * settings.scale - 2 * settings.min_margin

    if settings.divisible_by > 1:
        unit_latent = settings.divisible_by / vae_factor
        if unit_latent > 0:
            work_w = math.floor(work_w / unit_latent) * unit_latent
            work_h = math.floor(work_h / unit_latent) * unit_latent

    # Clamp to minimum 1.0 to avoid ZeroDivisionError in modular arithmetic
    # and nonsensical wrapping with negative dimensions.
    work_w = max(work_w, 1.0)
    work_h = max(work_h, 1.0)

    return work_w, work_h


def rect_tiling(
    x: int,
    y: int,
    original_size: tuple[int, int],
    padded_size: tuple[int, int],
    settings: Settings,
    vae_factor: int,
) -> tuple[int, int]:
    """Rectangular tiling with float-point dimensions.

    Uses float modular arithmetic for wrapping, with rounding only at final output.
    Pixels inside the float rectangle map to themselves.
    Pixels outside wrap to the opposite side.

    :param x: X coordinate in padded space
    :param y: Y coordinate in padded space
    :param original_size: (width, height) of original content
    :param padded_size: (width, height) of padded tensor
    :param settings: Tiling settings
    :param vae_factor: VAE downscale factor for divisible_by conversion
    :return: (new_x, new_y) source coordinates in padded space
    """
    ow, oh = original_size
    pw, ph = padded_size

    work_w, work_h = compute_float_rect_dims(ow, oh, settings, vae_factor)

    cx = pw / 2.0
    cy = ph / 2.0

    rel_x = x - cx
    rel_y = y - cy

    new_x = cx + ((rel_x + work_w / 2) % work_w) - work_w / 2
    new_y = cy + ((rel_y + work_h / 2) % work_h) - work_h / 2

    return (round(new_x), round(new_y))
