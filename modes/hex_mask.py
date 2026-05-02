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
def _build_inside_mask(width: int, height: int, rotation: float) -> torch.Tensor:
    """
    Build binary mask of pixels inside the hex.

    A pixel is inside if hex_tiling maps it to itself. Cached to avoid
    redundant computation when both border and neighbor masks are needed.

    :return: Bool tensor of shape (height, width)
    """
    from .hex_vectorized import hex_tiling_vectorized

    hex_size = min(width, height) // 2
    mapped_x, mapped_y = hex_tiling_vectorized(width, height, rotation, hex_size)

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
    inside = _build_inside_mask(width, height, settings.rotation)  # (H, W) bool
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


def _compute_sector_map_at(width: int, height: int, cx: float, cy: float) -> torch.Tensor:
    """
    Sector map centered on (cx, cy) instead of image center.

    Same sector划分 as _compute_sector_map but for offset tile positions.
    """
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij',
    )
    dx = xs - cx
    dy = cy - ys  # Flip Y for math coordinates
    angles = torch.atan2(dy, dx) % (2 * math.pi)
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
    inside = inside_mask if inside_mask is not None else _build_inside_mask(width, height, settings.rotation)
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
    inside = inside_mask if inside_mask is not None else _build_inside_mask(width, height, settings.rotation)
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
    inside = _build_inside_mask(width, height, settings.rotation)
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
    inside = _build_inside_mask(width, height, settings.rotation)
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


# Axial coordinate offsets for 6 neighbors of hex (0,0).
# Used with hex basis [[sqrt(3), sqrt(3)/2], [0, 3/2]] to compute pixel offsets.
_NEIGHBOR_AXIAL = {
    "E":  (1,  0),
    "NE": (1, -1),
    "NW": (0, -1),
    "W":  (-1, 0),
    "SW": (-1, 1),
    "SE": (0,  1),
}


def _neighbor_offset_px(direction: str, hex_radius: float) -> tuple[float, float]:
    """Pixel offset from center hex center to neighbor hex center."""
    q, r = _NEIGHBOR_AXIAL[direction]
    sqrt3 = math.sqrt(3)
    return (hex_radius * (sqrt3 * q + sqrt3 / 2 * r),
            hex_radius * (3.0 / 2 * r))


def compute_edge_buffer_mask(
    width: int,
    height: int,
    rotation: float,
    border_width: float,
    edge_buffer_depth: int,
    active_directions: set[int],
) -> torch.Tensor:
    """
    Compute per-direction boolean masks for the latent edge buffer zone.

    Identifies the outermost N pixels of the border ring per active direction.
    These pixels are hard-pasted with neighbor latent content and excluded from
    the inpaint mask, seeding the diffusion boundary with correct adjacent content.

    :param edge_buffer_depth: 0 = boundary only, N = N pixels deeper
    :param active_directions: Direction indices (0-5) that have a neighbour
    :return: Boolean tensor (6, H, W) per direction. True = buffer pixel.
    """
    inside = _build_inside_mask(width, height, rotation)
    hex_radius = min(width, height) // 2
    erosion = max(1, int(border_width * hex_radius))
    eroded = _erode_mask(inside, erosion)
    border = inside & ~eroded

    sectors = _compute_sector_map(width, height)
    dist_to_boundary = _manhattan_distance_to_region(~inside)
    threshold = edge_buffer_depth + 1

    buffer_masks = torch.zeros((6, height, width), dtype=torch.bool)
    for d in active_directions:
        sector_border = border & (sectors == d)
        buffer_masks[d] = sector_border & (torch.from_numpy(dist_to_boundary) <= threshold)

    return buffer_masks


def create_central_tile_masks(
    width: int,
    height: int,
    hex_radius: int,
    border_width: float = 0.2,
    feather_pixels: int = 0,
    active_directions: set[int] | None = None,
    masked_directions: set[int] | None = None,
    tile_priorities: list[int | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """
    Generate masks for CentralTile mode using hex-mask approach.

    Same approach as CornerEdges: build hex masks for center + each neighbor
    at offset positions, create tile_map for non-overlapping ownership, and
    mask only the lower-priority tile's border pixels.

    Overlap pixels at shared edges are assigned to the higher-priority tile
    in the tile_map, so they are automatically excluded from the mask.

    :param hex_radius: Hex circumradius in pixels
    :param active_directions: Direction indices (0-5) that have a neighbour
    :param masked_directions: Direction indices where neighbor has higher priority
    :param tile_priorities: [center_pri, E_pri, NE_pri, NW_pri, W_pri, SW_pri, SE_pri]
    :return: (border_mask (1,H,W), neighbor_masks (6,H,W), hex_masks list, tile_map (H,W))
    """
    if active_directions is None:
        active_directions = set(range(6))
    if masked_directions is None:
        masked_directions = active_directions

    cx, cy = width / 2.0, height / 2.0
    erosion = max(1, int(border_width * hex_radius))

    # Build hex masks: index 0=center, 1-6=neighbors by NEIGHBOR_DIRECTIONS order
    hex_masks = [_build_hex_mask_at(width, height, cx, cy, hex_radius)]
    for d in NEIGHBOR_DIRECTIONS:
        ox, oy = _neighbor_offset_px(d, hex_radius)
        hex_masks.append(_build_hex_mask_at(width, height, cx + ox, cy + oy, hex_radius))

    # Build tile_map: non-overlapping pixel ownership.
    # Priority: lower number = higher priority (e.g., 0=highest, 10=lowest).
    # Highest-priority tile processed first so it claims all overlap pixels.
    # Lower-priority tiles only get unclaimed pixels.
    # Result: overlap at shared edges is owned by the higher-priority tile.
    tile_map = torch.full((height, width), -1, dtype=torch.long)
    tile_order = list(range(7))
    if tile_priorities:
        tile_order.sort(key=lambda i: (
            tile_priorities[i] if i < len(tile_priorities) and tile_priorities[i] is not None
            else float('inf')
        ))
    for tile_idx in tile_order:
        unassigned = tile_map < 0
        tile_map[hex_masks[tile_idx] & unassigned] = tile_idx

    # Angular fallback for uncovered pixels
    uncovered = tile_map < 0
    if uncovered.any():
        sectors = _compute_sector_map(width, height)
        tile_map[uncovered] = sectors[uncovered] + 1  # +1 because 0=center

    # Erode center hex for border ring
    pad = erosion + 1
    padded = F.pad(
        hex_masks[0].float().unsqueeze(0).unsqueeze(0),
        [pad] * 4, mode='constant', value=1.0,
    )
    eroded_center = _erode_mask(padded.squeeze().bool(), erosion)
    if pad > 0:
        eroded_center = eroded_center[pad:-pad, pad:-pad]
    border_ring = hex_masks[0] & ~eroded_center

    # Sector map for directional filtering
    sectors = _compute_sector_map(width, height)

    # Per-direction neighbor masks (for visualization outputs)
    neighbor_masks = torch.zeros((6, height, width), dtype=torch.float32)
    for d in active_directions:
        neighbor_masks[d] = (border_ring & (sectors == d)).float()

    # Build border mask: center-owned border pixels in masked directions only.
    # tile_map == 0 excludes overlap pixels (assigned to higher-priority neighbors).
    border_mask = torch.zeros(height, width, dtype=torch.float32)

    if feather_pixels > 0:
        dist_to_eroded = torch.from_numpy(np.clip(
            _manhattan_distance_to_region(eroded_center) / feather_pixels, 0.0, 1.0,
        ))
    else:
        dist_to_eroded = torch.ones(height, width, dtype=torch.float32)

    for d in masked_directions:
        owns_center = tile_map == 0
        dir_border = border_ring & owns_center & (sectors == d)
        if dir_border.any():
            feathered = dir_border.float() * dist_to_eroded
            border_mask = torch.max(border_mask, feathered)

    return border_mask.unsqueeze(0), neighbor_masks, hex_masks, tile_map


def _build_hex_mask_at(
    width: int, height: int, cx: float, cy: float, circumradius: float,
) -> torch.Tensor:
    """
    Build boolean mask for a pointy-top hex at an arbitrary position.

    :param cx: Hex center X in image coordinates
    :param cy: Hex center Y in image coordinates (Y down)
    :param circumradius: Hex circumradius
    :return: Bool tensor (H, W)
    """
    apothem = circumradius * math.sqrt(3) / 2

    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij',
    )
    dx = xs - cx
    dy = cy - ys  # flip Y for math coords

    dist = torch.sqrt(dx * dx + dy * dy)
    angles = torch.atan2(dy, dx) % (2 * math.pi)

    sector_angles = (torch.round(angles / (math.pi / 3)) * (math.pi / 3)) % (2 * math.pi)
    local_angles = (angles - sector_angles + math.pi) % (2 * math.pi) - math.pi

    cos_local = torch.clamp(torch.cos(local_angles), min=1e-6)
    boundary_dist = apothem / cos_local

    return dist <= boundary_dist + 0.5  # half-pixel tolerance for boundary pixels


# Unit offsets for each corner's 3 tiles: (central, neighbor1, neighbor2).
# Multiply by hex_radius to get pixel offsets.
_CORNER_OFFSET_UNITS = {
    "N":  (( 0.0,          1.0), ( 0.866025404, -0.5), (-0.866025404, -0.5)),
    "NE": ((-0.866025404,  0.5), ( 0.866025404,  0.5), ( 0.0,         -1.0)),
    "SE": ((-0.866025404, -0.5), ( 0.0,          1.0), ( 0.866025404, -0.5)),
    "S":  (( 0.0,         -1.0), (-0.866025404,  0.5), ( 0.866025404,  0.5)),
    "SW": (( 0.866025404, -0.5), (-0.866025404, -0.5), ( 0.0,          1.0)),
    "NW": (( 0.866025404,  0.5), ( 0.0,         -1.0), (-0.866025404,  0.5)),
}

# Sector pairs for each corner's 3 edges: (tile_a, sector_a, tile_b, sector_b).
# Tiles: 0=center, 1=neighbor1, 2=neighbor2.
# Sectors: 0=E, 1=NE, 2=NW, 3=W, 4=SW, 5=SE (matching NEIGHBOR_DIRECTIONS).
_CORNER_EDGE_SECTORS = {
    "N":  [(0, 1, 1, 4), (0, 2, 2, 5), (1, 3, 2, 0)],
    "NE": [(0, 0, 1, 3), (0, 1, 2, 4), (1, 2, 2, 5)],
    "SE": [(0, 5, 1, 2), (0, 0, 2, 3), (1, 1, 2, 4)],
    "S":  [(0, 4, 1, 1), (0, 5, 2, 2), (1, 0, 2, 3)],
    "SW": [(0, 3, 1, 0), (0, 4, 2, 1), (1, 5, 2, 2)],
    "NW": [(0, 2, 1, 5), (0, 3, 2, 0), (1, 4, 2, 1)],
}

# Per-edge direction vectors (image coords, Y down) for extent projection.
# Each entry: [(edge0_dx, edge0_dy), (edge1_dx, edge1_dy), (edge2_dx, edge2_dy)]
# Edge N direction is along the hex edge from the shared vertex, perpendicular
# to the inter-center line, pointing away from the third tile.
_CORNER_EDGE_DIRECTIONS = {
    "N":  [( 0.866025404,  0.5), (-0.866025404,  0.5), ( 0.0,        -1.0)],
    "NE": [( 0.0,          1.0), (-0.866025404, -0.5), ( 0.866025404, -0.5)],
    "SE": [(-0.866025404,  0.5), ( 0.0,         -1.0), ( 0.866025404,  0.5)],
    "S":  [(-0.866025404, -0.5), ( 0.866025404, -0.5), ( 0.0,          1.0)],
    "SW": [( 0.0,         -1.0), ( 0.866025404,  0.5), (-0.866025404,  0.5)],
    "NW": [( 0.866025404, -0.5), ( 0.0,          1.0), (-0.866025404, -0.5)],
}

# Sector start angles for angular fallback (same values as corner_composite._CORNER_SECTOR_START)
_CORNER_SECTOR_STARTS = {
    "N":  math.radians(210),
    "NE": math.radians(150),
    "SE": math.radians(90),
    "S":  math.radians(30),
    "SW": math.radians(330),
    "NW": math.radians(270),
}


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

    Uses sector-based border rings (same approach as CentralTile mode):
    for each tile, compute hex mask -> erode -> sector-based border ring.
    Each edge selects the appropriate sector from each tile based on priority.
    """
    hex_radius = min(width, height) // 2
    erosion_pixels = max(1, int(border_width * hex_radius))

    if mask_extent == "half_edge":
        extent_pixels = round(hex_radius / 2)
    else:
        extent_pixels = hex_radius

    unit_offsets = _CORNER_OFFSET_UNITS[corner]
    float_offsets = [(ux * hex_radius, uy * hex_radius) for ux, uy in unit_offsets]

    # Build hex masks, eroded masks, and sector maps for each tile
    # Pad before erosion so image boundaries don't create false border rings
    pad = erosion_pixels + 1
    hex_masks = []
    eroded_full = []
    eroded_half = []
    sector_maps = []
    for ox, oy in float_offsets:
        cx = width / 2.0 + ox
        cy = height / 2.0 + oy
        inside = _build_hex_mask_at(width, height, cx, cy, hex_radius)
        hex_masks.append(inside)

        padded = F.pad(inside.float().unsqueeze(0).unsqueeze(0), [pad] * 4, mode='constant', value=1.0)
        eroded_padded_full = _erode_mask(padded.squeeze().bool(), erosion_pixels)
        eroded_full.append(eroded_padded_full[pad:-pad, pad:-pad] if pad > 0 else eroded_padded_full)

        half_erosion = max(1, erosion_pixels // 2)
        half_pad = half_erosion + 1
        padded_half = F.pad(inside.float().unsqueeze(0).unsqueeze(0), [half_pad] * 4, mode='constant', value=1.0)
        eroded_padded_half = _erode_mask(padded_half.squeeze().bool(), half_erosion)
        eroded_half.append(eroded_padded_half[half_pad:-half_pad, half_pad:-half_pad] if half_pad > 0 else eroded_padded_half)

        sector_maps.append(_compute_sector_map_at(width, height, cx, cy))

    # Pre-compute pixel coordinates relative to vertex for extent projection
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij',
    )
    vx, vy = width / 2.0, height / 2.0
    rel_x = xs - vx
    rel_y = ys - vy

    # Pre-compute distance transforms for feathering
    if feather_pixels > 0:
        dist_to_eroded_full = [
            torch.from_numpy(np.clip(
                _manhattan_distance_to_region(e) / feather_pixels, 0.0, 1.0,
            ))
            for e in eroded_full
        ]
        dist_to_eroded_half = [
            torch.from_numpy(np.clip(
                _manhattan_distance_to_region(e) / feather_pixels, 0.0, 1.0,
            ))
            for e in eroded_half
        ]

    # Non-overlapping tile assignment: each pixel belongs to exactly one tile.
    # Highest priority (lowest number) processed first so overlap pixels at
    # shared hex edges are assigned to the preserved tile, matching the
    # priority-ordered composite.
    tile_map = torch.full((height, width), -1, dtype=torch.long)
    tile_order = list(range(3))
    if tile_priorities and all(p is not None for p in tile_priorities):
        tile_order.sort(key=lambda i: tile_priorities[i])  # highest priority first
    for tile_idx in tile_order:
        unassigned = tile_map < 0
        tile_map[hex_masks[tile_idx] & unassigned] = tile_idx
    uncovered = tile_map < 0
    if uncovered.any():
        # 120-degree angular sectors from vertex for outer region
        vertex_angles = torch.atan2(-rel_y, rel_x) % (2 * math.pi)
        sector_start = _CORNER_SECTOR_STARTS[corner]
        rel_angle = (vertex_angles - sector_start) % (2 * math.pi)
        sector_fb = (rel_angle / (2 * math.pi / 3)).long() % 3
        tile_map[uncovered] = sector_fb[uncovered]

    edges = _CORNER_EDGE_SECTORS[corner]
    edge_dirs = _CORNER_EDGE_DIRECTIONS[corner]
    combined_mask = torch.zeros(height, width, dtype=torch.float32)

    for edge_idx, (tile_a, sec_a, tile_b, sec_b) in enumerate(edges):
        # Per-edge extent: project pixel positions onto edge direction
        dx, dy = edge_dirs[edge_idx]
        proj = rel_x * dx + rel_y * dy
        in_extent = (proj >= 0) & (proj <= extent_pixels)
        pri_a = tile_priorities[tile_a] if tile_priorities else None
        pri_b = tile_priorities[tile_b] if tile_priorities else None

        if pri_a is not None and pri_b is not None and pri_a == pri_b:
            continue

        # Use tile_map directly for non-overlapping tile ownership.
        # This excludes overlap pixels from the mask by design.
        owns_a = tile_map == tile_a
        owns_b = tile_map == tile_b
        if pri_a is None and pri_b is None:
            mask_a = owns_a & ~eroded_half[tile_a] & (sector_maps[tile_a] == sec_a) & in_extent
            mask_b = owns_b & ~eroded_half[tile_b] & (sector_maps[tile_b] == sec_b) & in_extent
            feather_a = dist_to_eroded_half[tile_a] if feather_pixels > 0 else None
            feather_b = dist_to_eroded_half[tile_b] if feather_pixels > 0 else None
        elif pri_a is not None and pri_b is not None:
            if pri_a > pri_b:
                mask_a = owns_a & ~eroded_full[tile_a] & (sector_maps[tile_a] == sec_a) & in_extent
                mask_b = torch.zeros(height, width, dtype=torch.bool)
                feather_a = dist_to_eroded_full[tile_a] if feather_pixels > 0 else None
                feather_b = None
            else:
                mask_a = torch.zeros(height, width, dtype=torch.bool)
                mask_b = owns_b & ~eroded_full[tile_b] & (sector_maps[tile_b] == sec_b) & in_extent
                feather_a = None
                feather_b = dist_to_eroded_full[tile_b] if feather_pixels > 0 else None
        elif pri_a is None:
            mask_a = owns_a & ~eroded_full[tile_a] & (sector_maps[tile_a] == sec_a) & in_extent
            mask_b = torch.zeros(height, width, dtype=torch.bool)
            feather_a = dist_to_eroded_full[tile_a] if feather_pixels > 0 else None
            feather_b = None
        else:
            mask_a = torch.zeros(height, width, dtype=torch.bool)
            mask_b = owns_b & ~eroded_full[tile_b] & (sector_maps[tile_b] == sec_b) & in_extent
            feather_a = None
            feather_b = dist_to_eroded_full[tile_b] if feather_pixels > 0 else None

        for mask, feather in [(mask_a, feather_a), (mask_b, feather_b)]:
            if mask.any():
                if feather_pixels > 0 and feather is not None:
                    edge_float = mask.float() * feather
                else:
                    edge_float = mask.float()
                combined_mask = torch.max(combined_mask, edge_float)

    return combined_mask.unsqueeze(0)
