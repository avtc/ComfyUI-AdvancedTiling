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


@functools.cache
def create_waste_mask(width: int, height: int, settings: Settings) -> torch.Tensor:
    """Create a mask where waste area (outside hex) = 1.0, inside hex = 0.0.

    Uses the same hex geometry as _build_inside_mask so the mask aligns
    exactly with the composited latent's waste regions.

    :param width: Width at the target resolution (latent or image).
    :param height: Height at the target resolution.
    :param settings: Tiling settings providing hex geometry.
    :return: Float tensor of shape (1, H, W) with values 0.0 (inside) or 1.0 (waste).
    """
    inside = _build_inside_mask(width, height, settings)  # (H, W) bool
    waste = (~inside).float().unsqueeze(0)  # (1, H, W)
    return waste


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


def _manhattan_distance_to_region(region: torch.Tensor) -> np.ndarray:
    """
    Manhattan distance from each pixel to the nearest True pixel in *region*.

    Two-pass raster scan. O(H*W).

    :param region: Bool tensor of shape (H, W)
    :return: Float64 array of shape (H, W), 0 inside *region*
    """
    dist = np.where(region.numpy(), np.float64(0), np.float64(np.inf))
    H, W = dist.shape

    for i in range(1, H):
        dist[i] = np.minimum(dist[i], dist[i - 1] + 1)
    for j in range(1, W):
        dist[:, j] = np.minimum(dist[:, j], dist[:, j - 1] + 1)
    for i in range(H - 2, -1, -1):
        dist[i] = np.minimum(dist[i], dist[i + 1] + 1)
    for j in range(W - 2, -1, -1):
        dist[:, j] = np.minimum(dist[:, j], dist[:, j + 1] + 1)

    return dist


def _diamond_erode(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """
    Diagonal (diamond SE) morphological erosion via Manhattan distance transform.

    :param mask: Bool tensor of shape (H, W)
    :param pixels: Erosion half-width (Manhattan distance threshold)
    :return: Eroded bool tensor
    """
    if pixels <= 0:
        return mask.clone()

    dist = _manhattan_distance_to_region(~mask)
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


def create_feathered_masks(
    width: int,
    height: int,
    settings: Settings,
    border_width: float = 0.2,
    feather_pixels: int = 0,
    feather_sides: bool = True,
    active_directions: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute inside, border, and neighbor masks with directional feathering.

    When *feather_pixels* > 0, applies a soft linear falloff on:

    - **Inner edges** (toward hex center): always feathered.
    - **Side edges**: feathered only when *feather_sides* is True AND the
      adjacent sector is *inactive*.
    - **Outer edges** (hex boundary): always sharp (value 1).

    Inactive directions produce all-zero masks.

    :param feather_pixels: Feather radius in pixels (0 = binary masks).
    :param feather_sides: If False, only inner edges are feathered;
                          side edges stay sharp regardless of neighbours.
    :param active_directions: Direction indices (0-5) that have a neighbour.
                              ``None`` means all active.
    :return: (inside_mask (H,W) bool, border_mask (1,H,W) float,
              neighbor_masks (6,H,W) float)
    """
    inside = _build_inside_mask(width, height, settings)
    hex_radius = min(width, height) // 2
    erosion_pixels = max(1, int(border_width * hex_radius))
    eroded = _erode_mask(inside, erosion_pixels)
    border = inside & ~eroded
    sectors = _compute_sector_map(width, height)

    if active_directions is None:
        active_directions = set(range(6))

    # Binary masks for active directions only
    neighbor_masks = torch.zeros((6, height, width), dtype=torch.float32)
    for d in active_directions:
        neighbor_masks[d] = (border & (sectors == d)).float()

    # Directional feathering
    if feather_pixels > 0 and active_directions:
        # Distance from each pixel to the inner (eroded) boundary
        dist_to_inner = _manhattan_distance_to_region(eroded)
        inner_weight = torch.from_numpy(
            np.clip(dist_to_inner / feather_pixels, 0.0, 1.0),
        )

        # Pre-compute angular data only when side feathering is enabled
        if feather_sides:
            ys, xs = torch.meshgrid(
                torch.arange(height, dtype=torch.float32),
                torch.arange(width, dtype=torch.float32),
                indexing='ij',
            )
            dx = xs - width / 2.0
            dy = -(ys - height / 2.0)  # math coords
            radius = torch.sqrt(dx * dx + dy * dy)
            offset_angles = (
                torch.atan2(dy, dx) % (2 * math.pi) + math.pi / 6
            ) % (2 * math.pi)

            # Distance to hex boundary for fading side feathering near outer edge
            dist_to_outer = _manhattan_distance_to_region(~inside)
            outer_weight = torch.from_numpy(
                np.clip(dist_to_outer / feather_pixels, 0.0, 1.0),
            )

        for d in active_directions:
            sector_mask = border & (sectors == d)
            if not sector_mask.any():
                continue

            feather = inner_weight

            if feather_sides:
                # Left boundary — shared with sector (d-1) % 6
                if (d - 1) % 6 not in active_directions:
                    ang = offset_angles - d * (math.pi / 3)
                    side_w = (ang * radius / feather_pixels).clamp(0, 1)
                    # Side feathering fades out near outer edge (outer_weight→0)
                    feather = feather * (1 - outer_weight * (1 - side_w))

                # Right boundary — shared with sector (d+1) % 6
                if (d + 1) % 6 not in active_directions:
                    ang = (d + 1) * (math.pi / 3) - offset_angles
                    side_w = (ang * radius / feather_pixels).clamp(0, 1)
                    feather = feather * (1 - outer_weight * (1 - side_w))

            neighbor_masks[d] = feather * sector_mask.float()

    border_mask = neighbor_masks.sum(dim=0, keepdim=True).clamp(0, 1)
    return inside, border_mask, neighbor_masks


def create_corner_masks(
    width: int,
    height: int,
    corner: str,
    border_width: float = 0.2,
    feather_pixels: int = 0,
    mask_extent: str = "half_edge",
    tile_priorities: list[int | None] | None = None,
) -> torch.Tensor:
    """
    Generate a combined border mask for corner inpainting.

    Creates masks along 3 edges where tiles meet at the corner,
    applying priority-based side selection.

    :param width: Image width
    :param height: Image height
    :param corner: Corner name ("N", "NE", "SE", "S", "SW", "NW")
    :param border_width: Border band width as fraction of hex radius
    :param feather_pixels: Feather radius in pixels (0 = sharp)
    :param mask_extent: "half_edge" (R/2 from center) or "full_edge" (R)
    :param tile_priorities: List of 3 priorities [center, n1, n2] or None.
        None means unknown. Used to decide which side(s) to mask.
    :return: Combined border mask (1, H, W) float
    """
    hex_radius = min(width, height) // 2
    erosion_pixels = max(1, int(border_width * hex_radius))

    # How far from center the mask extends along each edge
    if mask_extent == "half_edge":
        extent_pixels = round(hex_radius / 2)
    else:  # full_edge
        extent_pixels = hex_radius

    # Boundary angles (math coords) for each corner
    _CORNER_BOUNDARIES = {
        "N":  [math.radians(210), math.radians(330), math.radians(90)],
        "NE": [math.radians(270), math.radians(150), math.radians(30)],
        "SE": [math.radians(90),  math.radians(210), math.radians(330)],
        "S":  [math.radians(30),  math.radians(150), math.radians(270)],
        "SW": [math.radians(330), math.radians(90),  math.radians(210)],
        "NW": [math.radians(30),  math.radians(270), math.radians(150)],
    }

    boundaries = _CORNER_BOUNDARIES[corner]

    # tile_priorities: [center, n1, n2]
    # boundaries: [center↔n1, center↔n2, n1↔n2]
    # For each boundary, the two adjacent tiles are:
    #   boundary 0 (center↔n1): tiles 0 and 1
    #   boundary 1 (center↔n2): tiles 0 and 2
    #   boundary 2 (n1↔n2): tiles 1 and 2
    edge_tile_pairs = [(0, 1), (0, 2), (1, 2)]

    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij',
    )
    dx = xs - width / 2.0
    dy = -(ys - height / 2.0)  # math coords
    dist_from_center = torch.sqrt(dx * dx + dy * dy)
    pixel_angles = torch.atan2(dy, dx) % (2 * math.pi)

    combined_mask = torch.zeros(height, width, dtype=torch.float32)

    for edge_idx, (boundary_angle, (tile_a, tile_b)) in enumerate(
        zip(boundaries, edge_tile_pairs)
    ):
        # Determine which side(s) to mask based on priority
        pri_a = tile_priorities[tile_a] if tile_priorities else None
        pri_b = tile_priorities[tile_b] if tile_priorities else None

        if pri_a is not None and pri_b is not None and pri_a == pri_b:
            # Equal priority: no mask needed (seamless tiles)
            continue

        # Angular distance to boundary line
        angle_diff = (pixel_angles - boundary_angle + math.pi) % (2 * math.pi) - math.pi
        # Perpendicular distance from boundary line
        perp_dist = torch.abs(torch.sin(angle_diff)) * dist_from_center

        # Distance along boundary from center
        along_dist = torch.abs(torch.cos(angle_diff)) * dist_from_center

        # Mask conditions
        in_extent = along_dist <= extent_pixels
        in_band = perp_dist <= erosion_pixels

        if pri_a is None and pri_b is None:
            # Both unknown: mask both sides equally
            mask_band = in_extent & in_band
        elif pri_a is not None and pri_b is not None:
            # Different priorities: mask lower-priority side only
            # Lower priority = higher index value
            if pri_a > pri_b:
                # Mask tile A's side (positive angle_diff)
                side_mask = angle_diff > 0
            else:
                # Mask tile B's side (negative angle_diff)
                side_mask = angle_diff < 0
            mask_band = in_extent & in_band & side_mask
        elif pri_a is None:
            # A unknown, B known: mask A's side only
            side_mask = angle_diff > 0
            mask_band = in_extent & in_band & side_mask
        else:
            # B unknown, A known: mask B's side only
            side_mask = angle_diff < 0
            mask_band = in_extent & in_band & side_mask

        # Apply feathering
        if feather_pixels > 0 and mask_band.any():
            feather_weight = torch.clamp(
                (erosion_pixels - perp_dist) / feather_pixels, 0.0, 1.0
            )
            edge_mask = mask_band.float() * feather_weight
        else:
            edge_mask = mask_band.float()

        combined_mask = torch.max(combined_mask, edge_mask)

    return combined_mask.unsqueeze(0)  # (1, H, W)
