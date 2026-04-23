"""
Hex border mask generation with per-neighbor segmentation.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from . import Settings
from .hex import hex_tiling


# Neighbor directions for pointy-top hexagon (clockwise from East)
NEIGHBOR_DIRECTIONS = ["E", "NE", "NW", "W", "SW", "SE"]


def _build_inside_mask(width: int, height: int, settings: Settings) -> torch.Tensor:
    """
    Build binary mask of pixels inside the hex.

    A pixel is inside if hex_tiling maps it to itself.

    :return: Bool tensor of shape (height, width)
    """
    mask = torch.zeros((height, width), dtype=torch.bool)
    for y in range(height):
        for x in range(width):
            nx, ny = hex_tiling(x, y, (width, height), (width, height), settings)
            if nx == x and ny == y:
                mask[y, x] = True
    return mask


def _erode_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """
    Erode a binary mask by the given number of pixels using max_pool.

    :param mask: Bool tensor of shape (H, W)
    :param pixels: Erosion radius in pixels
    :return: Eroded bool tensor of shape (H, W)
    """
    if pixels <= 0:
        return mask.clone()

    kernel_size = 2 * pixels + 1
    # Use min_pool (negated max_pool) for erosion
    fmask = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    # Erosion = -max_pool(-mask) with appropriate padding
    padded = F.pad(fmask, [pixels, pixels, pixels, pixels], mode='constant', value=0)
    eroded = -F.max_pool2d(-padded, kernel_size, stride=1)
    return eroded.squeeze(0).squeeze(0) > 0.5


def _pixel_angle_from_center(
    x: int, y: int, width: int, height: int
) -> float:
    """
    Compute angle of pixel from image center in radians [0, 2*pi).
    """
    dx = x - width / 2.0
    dy = -(y - height / 2.0)  # Flip Y for standard math coordinates
    return math.atan2(dy, dx) % (2 * math.pi)


def _angle_to_direction(angle: float) -> int:
    """
    Map angle [0, 2*pi) to neighbor direction index 0-5.

    For pointy-top hex, sectors are centered on edge midpoints (0°, 60°, ...)
    with boundaries at vertices (30°, 90°, 150°, ...). The 30° offset aligns
    sector boundaries with hex vertices.

      E  = 0°     (index 0, sector 330°-30°)
      NE = 60°    (index 1, sector 30°-90°)
      NW = 120°   (index 2, sector 90°-150°)
      W  = 180°   (index 3, sector 150°-210°)
      SW = 240°   (index 4, sector 210°-270°)
      SE = 300°   (index 5, sector 270°-330°)
    """
    sector = int((angle + math.pi / 6) / (math.pi / 3)) % 6
    return sector


def create_border_mask(
    width: int, height: int, settings: Settings, border_width: float = 0.2
) -> torch.Tensor:
    """
    Generate annular border mask for hex inpainting.

    The mask marks pixels that are inside the hex and within border_width
    of the hex edge. Uses morphological erosion to find the border region.

    :param width: Image width
    :param height: Image height
    :param settings: Tiling settings
    :param border_width: Fraction of hex radius for border (0.05-0.45)
    :return: Float tensor of shape (1, height, width) with values 0-1
    """
    inside = _build_inside_mask(width, height, settings)
    hex_radius = min(width, height) // 2
    erosion_pixels = max(1, int(border_width * hex_radius))

    eroded = _erode_mask(inside, erosion_pixels)
    border = inside & ~eroded

    return border.float().unsqueeze(0)


def create_neighbor_masks(
    width: int, height: int, settings: Settings, border_width: float = 0.2
) -> torch.Tensor:
    """
    Generate 6 separate border masks, one per neighbor direction.

    Each mask covers a 60-degree wedge of the border ring.
    Non-overlapping — each border pixel belongs to exactly one neighbor.

    :param width: Image width
    :param height: Image height
    :param settings: Tiling settings
    :param border_width: Fraction of hex radius for border (0.05-0.45)
    :return: Float tensor of shape (6, height, width)
    """
    inside = _build_inside_mask(width, height, settings)
    hex_radius = min(width, height) // 2
    erosion_pixels = max(1, int(border_width * hex_radius))

    eroded = _erode_mask(inside, erosion_pixels)
    border = inside & ~eroded

    masks = torch.zeros((6, height, width), dtype=torch.float32)

    # Vectorized angle computation for border pixels only
    border_ys, border_xs = torch.where(border)
    for idx in range(len(border_xs)):
        x = border_xs[idx].item()
        y = border_ys[idx].item()
        angle = _pixel_angle_from_center(x, y, width, height)
        direction = _angle_to_direction(angle)
        masks[direction, y, x] = 1.0

    return masks
