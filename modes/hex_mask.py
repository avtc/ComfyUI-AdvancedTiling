"""
Hex border mask generation with per-neighbor segmentation.
"""

import math
import functools
import logging

import torch
import torch.nn.functional as F
import numpy as np
from . import Settings

logger = logging.getLogger("ComfyUI-AdvancedTiling")

# Neighbor directions for pointy-top hexagon (clockwise from East)
NEIGHBOR_DIRECTIONS = ["E", "NE", "NW", "W", "SW", "SE"]


@functools.cache
def _build_inside_mask(width: int, height: int, settings: Settings) -> torch.Tensor:
    """
    Build binary mask of pixels inside the hex.

    A pixel is inside if hex_tiling maps it to itself. Cached to avoid
    redundant computation when both border and neighbor masks are needed.

    :return: Bool tensor of shape (height, width)
    """
    from .hex import hex_tiling_vectorized

    mapped_x, mapped_y = hex_tiling_vectorized(width, height, settings)

    xs = np.arange(width, dtype=np.int64)[np.newaxis, :]
    ys = np.arange(height, dtype=np.int64)[:, np.newaxis]

    is_inside = (mapped_x == xs) & (mapped_y == ys)
    return torch.from_numpy(is_inside)


def _square_erode(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """
    Axis-aligned (square SE) morphological erosion via separable max_pool1d.

    :param mask: Bool tensor of shape (H, W)
    :param pixels: Erosion half-width
    :return: Eroded bool tensor
    """
    if pixels <= 0:
        return mask.clone()

    k = 2 * pixels + 1
    fmask = mask.float()

    h_input = fmask.unsqueeze(1)
    h_padded = F.pad(h_input, [pixels, pixels], mode='constant', value=0)
    h_eroded = -F.max_pool1d(-h_padded, k, stride=1)

    v_input = h_eroded.squeeze(1).t().contiguous().unsqueeze(1)
    v_padded = F.pad(v_input, [pixels, pixels], mode='constant', value=0)
    v_eroded = -F.max_pool1d(-v_padded, k, stride=1)

    return v_eroded.squeeze(1).t().contiguous() > 0.5


def _diamond_erode(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """
    Diagonal (diamond SE) morphological erosion via Manhattan distance transform.

    Computes L1 distance from each pixel to the nearest boundary using two
    forward/backward raster scans. O(H*W) total.

    :param mask: Bool tensor of shape (H, W)
    :param pixels: Erosion half-width (Manhattan distance threshold)
    :return: Eroded bool tensor
    """
    if pixels <= 0:
        return mask.clone()

    dist = np.where(mask.numpy(), np.float64(np.inf), 0.0)
    H, W = dist.shape

    for i in range(1, H):
        dist[i] = np.minimum(dist[i], dist[i - 1] + 1)
    for j in range(1, W):
        dist[:, j] = np.minimum(dist[:, j], dist[:, j - 1] + 1)
    for i in range(H - 2, -1, -1):
        dist[i] = np.minimum(dist[i], dist[i + 1] + 1)
    for j in range(W - 2, -1, -1):
        dist[:, j] = np.minimum(dist[:, j], dist[:, j + 1] + 1)

    return torch.from_numpy(dist >= pixels)


def _erode_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """
    Isotropic morphological erosion using octagonal structuring element.

    Combines square (axis-aligned) and diamond (diagonal) erosions.
    The square SE erodes by e/√2 along axes; the diamond SE erodes by e
    along diagonals. Their intersection approximates a circular SE with
    ±3.4% variation — far better than square-only erosion (±36.6%).

    :param mask: Bool tensor of shape (H, W)
    :param pixels: Erosion radius in pixels
    :return: Eroded bool tensor of shape (H, W)
    """
    if pixels <= 0:
        return mask.clone()

    e_sq = max(1, round(pixels / math.sqrt(2)))
    return _square_erode(mask, e_sq) & _diamond_erode(mask, pixels)


def _compute_sector_map(width: int, height: int) -> torch.Tensor:
    """
    Vectorized computation of angular sector for each pixel.

    :return: LongTensor of shape (height, width) with sector indices 0-5
    """
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij',
    )
    dx = xs - width / 2.0
    dy = -(ys - height / 2.0)  # Flip Y for math coordinates
    angles = torch.atan2(dy, dx) % (2 * math.pi)
    # 30° offset aligns sector boundaries with hex vertices
    return ((angles + math.pi / 6) / (math.pi / 3)).long() % 6


def create_border_mask(
    width: int, height: int, settings: Settings, border_width: float = 0.2,
    inside_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Generate annular border mask for hex inpainting.

    :param width: Image width
    :param height: Image height
    :param settings: Tiling settings
    :param border_width: Fraction of hex radius for border (0.05-0.45)
    :param inside_mask: Pre-computed inside mask (H, W) bool, computed if not provided
    :return: Float tensor of shape (1, height, width) with values 0-1
    """
    inside = inside_mask if inside_mask is not None else _build_inside_mask(width, height, settings)
    hex_radius = min(width, height) // 2
    erosion_pixels = max(1, int(border_width * hex_radius))

    eroded = _erode_mask(inside, erosion_pixels)
    border = inside & ~eroded

    return border.float().unsqueeze(0)


def create_neighbor_masks(
    width: int, height: int, settings: Settings, border_width: float = 0.2,
    inside_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Generate 6 separate border masks, one per neighbor direction.

    Each mask covers a 60-degree wedge of the border ring.
    Non-overlapping — each border pixel belongs to exactly one neighbor.

    :param width: Image width
    :param height: Image height
    :param settings: Tiling settings
    :param border_width: Fraction of hex radius for border (0.05-0.45)
    :param inside_mask: Pre-computed inside mask (H, W) bool, computed if not provided
    :return: Float tensor of shape (6, height, width)
    """
    inside = inside_mask if inside_mask is not None else _build_inside_mask(width, height, settings)
    hex_radius = min(width, height) // 2
    erosion_pixels = max(1, int(border_width * hex_radius))

    eroded = _erode_mask(inside, erosion_pixels)
    border = inside & ~eroded

    sectors = _compute_sector_map(width, height)

    masks = torch.zeros((6, height, width), dtype=torch.float32)
    for d in range(6):
        masks[d] = (border & (sectors == d)).float()

    return masks


def create_masks(
    width: int, height: int, settings: Settings, border_width: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute inside mask, border mask, and neighbor masks in one pass.

    Avoids redundant inside mask computation.

    :return: (inside_mask (H,W) bool, border_mask (1,H,W) float, neighbor_masks (6,H,W) float)
    """
    inside = _build_inside_mask(width, height, settings)
    hex_radius = min(width, height) // 2
    erosion_pixels = max(1, int(border_width * hex_radius))

    eroded = _erode_mask(inside, erosion_pixels)
    border = inside & ~eroded

    sectors = _compute_sector_map(width, height)

    border_mask = border.float().unsqueeze(0)
    neighbor_masks = torch.zeros((6, height, width), dtype=torch.float32)
    for d in range(6):
        neighbor_masks[d] = (border & (sectors == d)).float()

    return inside, border_mask, neighbor_masks
