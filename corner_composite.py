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

# Sector start angles for angular fallback.
# Computed from the boundary just clockwise of the central tile's center
# angle as seen from the shared vertex.
_CORNER_SECTOR_START = {
    "N":  math.radians(210),
    "NE": math.radians(150),
    "SE": math.radians(90),
    "S":  math.radians(30),
    "SW": math.radians(330),
    "NW": math.radians(270),
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


def _build_offset_hex_mask(
    width: int, height: int, ox: float, oy: float, hex_radius: float,
) -> torch.Tensor:
    """
    Build boolean mask for a pointy-top hex at offset position in output space.

    :param ox: X offset in image coordinates
    :param oy: Y offset in image coordinates
    :param hex_radius: Hex circumradius
    :return: Bool tensor (H, W)
    """
    cx = width / 2.0 + ox
    cy = height / 2.0 + oy
    apothem = hex_radius * math.sqrt(3) / 2

    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij',
    )
    dx = xs - cx
    dy = cy - ys  # flip Y for math coords

    dist = torch.sqrt(dx * dx + dy * dy)
    angles = torch.atan2(dy, dx) % (2 * math.pi)

    # Nearest edge-midpoint direction (0, 60, 120, 180, 240, 300 degrees)
    sector_angles = (torch.round(angles / (math.pi / 3)) * (math.pi / 3)) % (2 * math.pi)
    local_angles = (angles - sector_angles + math.pi) % (2 * math.pi) - math.pi

    cos_local = torch.clamp(torch.cos(local_angles), min=1e-6)
    boundary_dist = apothem / cos_local

    return dist <= boundary_dist + 0.5  # half-pixel tolerance for boundary pixels


def build_corner_tile_map(
    width: int, height: int, corner: str, hex_radius: float,
) -> torch.Tensor:
    """
    Build a map assigning each output pixel to one of 3 tiles.

    Uses hex masks (with sub-pixel offsets) for accurate tile boundaries.
    Falls back to 120-degree angular sectors for any pixels not covered
    by a hex mask (outer regions beyond circumradius).

    :param width: Output width
    :param height: Output height
    :param corner: Corner name
    :param hex_radius: Hex circumradius in pixels
    :return: LongTensor (H, W) with values 0 (central), 1 (neighbor1), 2 (neighbor2)
    """
    unit_offsets = _CORNER_OFFSETS[corner]
    # Float offsets for accurate hex mask boundaries
    float_offsets = [(ux * hex_radius, uy * hex_radius) for ux, uy in unit_offsets]

    # Primary: hex-mask-based assignment (accurate boundaries)
    tile_map = torch.full((height, width), -1, dtype=torch.long)
    for tile_idx, (ox, oy) in enumerate(float_offsets):
        hex_mask = _build_offset_hex_mask(width, height, ox, oy, hex_radius)
        unassigned = tile_map < 0
        tile_map[hex_mask & unassigned] = tile_idx

    # Fallback: angular sectors for uncovered pixels (outer region)
    uncovered = tile_map < 0
    if uncovered.any():
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing='ij',
        )
        dx = xs - width / 2.0
        dy = -(ys - height / 2.0)
        angles = torch.atan2(dy, dx) % (2 * math.pi)

        sector_start = _CORNER_SECTOR_START[corner]
        rel_angle = (angles - sector_start) % (2 * math.pi)
        sector_map = (rel_angle / (2 * math.pi / 3)).long() % 3
        tile_map[uncovered] = sector_map[uncovered]

    return tile_map


def _composite_priority_ordered(
    tiles, corner, hex_radius_px, tile_priorities=None, is_latent=True,
):
    """
    Composite 3 tiles by pasting in priority order.

    Lowest priority (highest number) is pasted first; highest priority
    (lowest number) is pasted last, overwriting at overlapping hex edges.
    This naturally places correct content at tile boundaries without needing
    a separate edge buffer.

    :param tiles: List of 3 tensors (latent 4D or image 4D)
    :param corner: Corner name
    :param hex_radius_px: Hex circumradius in pixels
    :param tile_priorities: List of 3 priorities (lower = higher priority), or None
    :param is_latent: True for latent (B,C,H,W), False for image (B,H,W,C)
    :return: Composited tensor
    """
    unit_offsets = _CORNER_OFFSETS[corner]
    float_offsets = [(ux * hex_radius_px, uy * hex_radius_px) for ux, uy in unit_offsets]
    int_offsets = get_corner_offsets(corner, hex_radius_px)

    if is_latent:
        B, C, H, W = tiles[0].shape
    else:
        B, H, W, C = tiles[0].shape

    # Build hex masks for each tile
    hex_masks = []
    for ox, oy in float_offsets:
        hm = _build_offset_hex_mask(W, H, ox, oy, hex_radius_px)
        hex_masks.append(hm)

    # Sort: lowest priority (highest number) first → highest priority last
    order = list(range(3))
    if tile_priorities and all(p is not None for p in tile_priorities):
        order.sort(key=lambda i: -tile_priorities[i])

    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.long),
        torch.arange(W, dtype=torch.long),
        indexing='ij',
    )

    result = torch.zeros_like(tiles[0])

    for tile_idx in order:
        ox, oy = int_offsets[tile_idx]
        src = tiles[tile_idx]
        mask = hex_masks[tile_idx]
        if not mask.any():
            continue
        src_ys = (ys - oy).clamp(0, H - 1)
        src_xs = (xs - ox).clamp(0, W - 1)
        if is_latent:
            result[:, :, mask] = src[:, :, src_ys[mask], src_xs[mask]]
        else:
            result[0, mask] = src[0, src_ys[mask], src_xs[mask]]

    # Angular fallback for uncovered pixels (outer region beyond circumradius)
    uncovered = ~hex_masks[0] & ~hex_masks[1] & ~hex_masks[2]
    if uncovered.any():
        tile_map_fb = build_corner_tile_map(W, H, corner, hex_radius_px)
        for tile_idx in order:
            tile_unc = uncovered & (tile_map_fb == tile_idx)
            if not tile_unc.any():
                continue
            ox, oy = int_offsets[tile_idx]
            src = tiles[tile_idx]
            src_ys = (ys - oy).clamp(0, H - 1)
            src_xs = (xs - ox).clamp(0, W - 1)
            if is_latent:
                result[:, :, tile_unc] = src[:, :, src_ys[tile_unc], src_xs[tile_unc]]
            else:
                result[0, tile_unc] = src[0, src_ys[tile_unc], src_xs[tile_unc]]

    return result


def composite_corner_latents(
    center_latent: torch.Tensor,
    neighbor1_latent: torch.Tensor,
    neighbor2_latent: torch.Tensor,
    corner: str,
    hex_radius_px: float,
    tile_priorities: list | None = None,
) -> torch.Tensor:
    """
    Composite 3 tile latents into a corner view with shared vertex at center.

    Pastes in priority order so higher-priority tiles overwrite at overlapping
    edges, eliminating the need for a separate edge buffer.

    :param tile_priorities: List of 3 priorities (lower = higher priority), or None
    """
    from .hex_inpaint import _normalize_latent

    center_4d, center_orig = _normalize_latent(center_latent)
    n1_4d, _ = _normalize_latent(neighbor1_latent)
    n2_4d, _ = _normalize_latent(neighbor2_latent)

    result = _composite_priority_ordered(
        [center_4d, n1_4d, n2_4d], corner, hex_radius_px,
        tile_priorities=tile_priorities, is_latent=True,
    )

    if len(center_orig) > 4:
        result = result.reshape(center_orig)
    return result


def composite_corner_preview(
    center_image: torch.Tensor,
    neighbor1_image: torch.Tensor,
    neighbor2_image: torch.Tensor,
    corner: str,
    tile_priorities: list | None = None,
) -> torch.Tensor:
    """
    Compose 3 tile images into a corner preview at pixel resolution.

    :param tile_priorities: List of 3 priorities (lower = higher priority), or None
    :return: Preview image (1, H, W, 3)
    """
    H, W = center_image.shape[1], center_image.shape[2]
    hex_radius = min(W, H) // 2

    return _composite_priority_ordered(
        [center_image, neighbor1_image, neighbor2_image],
        corner, hex_radius,
        tile_priorities=tile_priorities, is_latent=False,
    )
