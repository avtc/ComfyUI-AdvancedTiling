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
        ys, xs = torch.meshgrid(
            torch.arange(H, dtype=torch.long),
            torch.arange(W, dtype=torch.long),
            indexing='ij',
        )
        src_ys = (ys - oy).clamp(0, H - 1)
        src_xs = (xs - ox).clamp(0, W - 1)

        if mask.any():
            result[:, :, mask] = src[:, :, src_ys[mask], src_xs[mask]]

    if len(center_orig) > 4:
        result = result.reshape(center_orig)
    return result


# Vertical edge direction: True = up, False = down
_VERT_EDGE_UP = {"N": True, "NE": False, "SE": True, "S": False, "SW": True, "NW": False}


def composite_corner_preview(
    center_image: torch.Tensor,
    neighbor1_image: torch.Tensor,
    neighbor2_image: torch.Tensor,
    corner: str,
) -> torch.Tensor:
    """
    Compose 3 tile images into a corner preview, cropped to remove border artifacts.

    Returns a square crop centered on the corner region, with the vertical hex
    edge ending at the image boundary.
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
        src_ys = (ys - oy).clamp(0, H - 1)
        src_xs = (xs - ox).clamp(0, W - 1)

        if mask.any():
            result[0, mask] = src[0, src_ys[mask], src_xs[mask]]

    # Geometric crop
    crop_size = round(hex_radius * math.sqrt(3))
    cx, cy = W / 2.0, H / 2.0

    half_w = crop_size / 2.0
    c0 = max(0, int(cx - half_w))
    c1 = min(W, int(cx + half_w))

    if _VERT_EDGE_UP[corner]:
        r0 = max(0, int(cy - hex_radius))
        r1 = r0 + crop_size
        r1 = min(H, r1)
    else:
        r1 = min(H, int(cy + hex_radius))
        r0 = r1 - crop_size
        r0 = max(0, r0)

    result = result[:, r0:r1, c0:c1, :]

    # Ensure square
    _, rh, rw, _ = result.shape
    if rh != rw:
        side = min(rh, rw)
        result = result[:, :side, :side, :]

    return result
