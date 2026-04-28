"""
Corner composition for hex tile CornerEdges mode.
Composites 3 tile latents with the shared corner at the output center.
"""

import math
import logging

import torch

logger = logging.getLogger("ComfyUI-AdvancedTiling")

# Corner names and their associated neighbor directions
CORNER_NAMES = ["N", "NE", "SE", "S", "SW", "NW"]

# Maps corner name to (neighbor_1_direction, neighbor_2_direction)
CORNER_NEIGHBORS = {
    "N":  ("NE", "NW"),
    "NE": ("E",  "NE"),
    "SE": ("SE", "E"),
    "S":  ("SW", "SE"),
    "SW": ("W",  "SW"),
    "NW": ("NW", "W"),
}

# Tile offsets (dx, dy) in pixels for each corner.
# Each row: (central_offset, neighbor1_offset, neighbor2_offset)
# R = hex circumradius. Offsets shift each tile so the shared vertex
# lands at the output image center (S/2, S/2).
# Derived from pointy-top hex vertex positions.
_CORNER_OFFSETS = {
    "N":  (( 0.0,          1.0), ( 0.866025404, -0.5), (-0.866025404, -0.5)),
    "NE": ((-0.866025404,  0.5), ( 0.866025404,  0.5), ( 0.0,         -1.0)),
    "SE": ((-0.866025404, -0.5), ( 0.0,          1.0), ( 0.866025404, -0.5)),
    "S":  (( 0.0,         -1.0), (-0.866025404,  0.5), ( 0.866025404,  0.5)),
    "SW": (( 0.866025404, -0.5), (-0.866025404, -0.5), ( 0.0,          1.0)),
    "NW": (( 0.866025404,  0.5), ( 0.0,         -1.0), (-0.866025404,  0.5)),
}


def get_corner_offsets(corner: str, hex_radius: float):
    """
    Get pixel offsets for the 3 tiles in a corner composition.

    :param corner: Corner name ("N", "NE", etc.)
    :param hex_radius: Hex circumradius in pixels
    :return: Tuple of 3 (dx, dy) offsets: (central, neighbor1, neighbor2)
    """
    unit_offsets = _CORNER_OFFSETS[corner]
    return tuple(
        (round(ux * hex_radius), round(uy * hex_radius))
        for ux, uy in unit_offsets
    )


def build_corner_tile_map(
    width: int, height: int, corner: str, hex_radius: float,
) -> torch.Tensor:
    """
    Build a map assigning each output pixel to one of 3 tiles.

    Uses 120-degree sectors radiating from the output center. Each sector
    is assigned to the tile whose content covers that angular range.

    :return: LongTensor (H, W) with values 0 (central), 1 (neighbor1), 2 (neighbor2)
    """
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij',
    )
    dx = xs - width / 2.0
    dy = -(ys - height / 2.0)  # flip Y for math coords
    angles = torch.atan2(dy, dx) % (2 * math.pi)

    # The 3 boundaries are at 120-degree intervals.
    # The starting angle depends on the corner.
    # Sector 0 (central tile) starts at boundary_0 and spans 120 degrees.
    # boundary_0 is the edge between central and neighbor1.
    # boundary_1 is between neighbor1 and neighbor2.
    # boundary_2 is between neighbor2 and central.
    _CORNER_SECTOR_START = {
        "N":  math.radians(210),
        "NE": math.radians(270),
        "SE": math.radians(90),
        "S":  math.radians(30),
        "SW": math.radians(330),
        "NW": math.radians(30),
    }

    sector_start = _CORNER_SECTOR_START[corner]
    # Relative angle from sector start, in [0, 2*pi)
    rel_angle = (angles - sector_start) % (2 * math.pi)
    # Tile index: 0=central (0-120), 1=neighbor1 (120-240), 2=neighbor2 (240-360)
    tile_map = (rel_angle / (2 * math.pi / 3)).long() % 3

    return tile_map


def composite_corner_latents(
    center_latent: torch.Tensor,
    neighbor1_latent: torch.Tensor,
    neighbor2_latent: torch.Tensor,
    corner: str,
    hex_radius_px: float,
) -> torch.Tensor:
    """
    Composite 3 tile latents into a corner view with shared vertex at center.

    :param center_latent: Central tile latent (1, C, H, W)
    :param neighbor1_latent: First neighbor latent
    :param neighbor2_latent: Second neighbor latent
    :param corner: Corner name
    :param hex_radius_px: Hex circumradius in pixels at latent resolution
    :return: Composited latent (same shape as inputs)
    """
    from .hex_inpaint import _normalize_latent

    center_4d, center_orig = _normalize_latent(center_latent)
    n1_4d, _ = _normalize_latent(neighbor1_latent)
    n2_4d, _ = _normalize_latent(neighbor2_latent)

    B, C, H, W = center_4d.shape
    offsets = get_corner_offsets(corner, hex_radius_px)

    result = torch.zeros_like(center_4d)
    tile_map = build_corner_tile_map(W, H, corner, hex_radius_px)

    latents = [center_4d, n1_4d, n2_4d]
    for tile_idx, (ox, oy) in enumerate(offsets):
        src = latents[tile_idx]
        mask = (tile_map == tile_idx)
        # For each pixel assigned to this tile, look up the source pixel
        # accounting for the offset
        ys, xs = torch.meshgrid(
            torch.arange(H, dtype=torch.long),
            torch.arange(W, dtype=torch.long),
            indexing='ij',
        )
        src_ys = ys - oy
        src_xs = xs - ox
        valid = mask & (src_ys >= 0) & (src_ys < H) & (src_xs >= 0) & (src_xs < W)

        if valid.any():
            result[:, :, valid] = src[:, :, src_ys[valid], src_xs[valid]]

    if len(center_orig) > 4:
        result = result.reshape(center_orig)
    return result


def composite_corner_preview(
    center_image: torch.Tensor,
    neighbor1_image: torch.Tensor,
    neighbor2_image: torch.Tensor,
    corner: str,
) -> torch.Tensor:
    """
    Compose 3 tile images into a corner preview at pixel resolution.

    :return: Preview image (1, H, W, 3)
    """
    H, W = center_image.shape[1], center_image.shape[2]
    hex_radius = min(W, H) // 2
    offsets = get_corner_offsets(corner, hex_radius)

    result = torch.zeros_like(center_image)
    tile_map = build_corner_tile_map(W, H, corner, hex_radius)

    images = [center_image, neighbor1_image, neighbor2_image]
    for tile_idx, (ox, oy) in enumerate(offsets):
        src = images[tile_idx]
        mask = (tile_map == tile_idx)
        ys, xs = torch.meshgrid(
            torch.arange(H, dtype=torch.long),
            torch.arange(W, dtype=torch.long),
            indexing='ij',
        )
        src_ys = ys - oy
        src_xs = xs - ox
        valid = mask & (src_ys >= 0) & (src_ys < H) & (src_xs >= 0) & (src_xs < W)

        if valid.any():
            result[0, valid] = src[0, src_ys[valid], src_xs[valid]]

    return result
