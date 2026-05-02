"""
CentralTile ASCII visualization using hex-mask approach (same as CornerEdges).
Builds actual hex masks for center + each neighbor at offset positions.
Creates tile_map to resolve overlaps at shared edges.
a=center, b=neighbor1, c=neighbor2, ... x/y = overwritten at shared edges
uppercase=masked, lowercase=preserved
"""

import sys
import os
import math
import torch

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_pkg_dir)
sys.path.insert(0, _parent)

import importlib
_pkg_name = os.path.basename(_pkg_dir)
_pkg = importlib.import_module(_pkg_name)
sys.modules[_pkg_name] = _pkg
sys.modules["ComfyUI_AdvancedTiling"] = _pkg

from modes.hex_mask import (
    NEIGHBOR_DIRECTIONS, _build_hex_mask_at, _compute_sector_map_at,
    _erode_mask, _manhattan_distance_to_region, _compute_sector_map,
)
import torch.nn.functional as F
import numpy as np


# Axial coordinate offsets for 6 neighbors of hex (0,0)
# Using basis [[sqrt(3), sqrt(3)/2], [0, 3/2]] * size
# hex_to_pixel(q,r) = size * (sqrt(3)*q + sqrt(3)/2*r, 3/2*r)
_AXIAL_OFFSETS = {
    "E":  (1,  0),
    "NE": (1, -1),
    "NW": (0, -1),
    "W":  (-1, 0),
    "SW": (-1, 1),
    "SE": (0,  1),
}


def _neighbor_pixel_offset(q, r, hex_radius):
    """Pixel offset from center hex to neighbor hex center."""
    sqrt3 = math.sqrt(3)
    ox = hex_radius * (sqrt3 * q + sqrt3 / 2 * r)
    oy = hex_radius * (3.0 / 2 * r)
    return (ox, oy)


def visualize():
    W, H = 166, 166
    border_width = 0.15
    hex_radius = min(W, H) // 2  # 83
    cx, cy = W / 2.0, H / 2.0

    # Priorities: center_pri=2
    center_pri = 2
    neighbor_priorities = {"E": 1, "NE": 0, "W": 3, "SE": 2}
    active_directions = {"E", "NE", "W", "SE"}

    # Masked: only where neighbor pri < center pri
    masked_directions = set()
    for d in active_directions:
        n_pri = neighbor_priorities[d]
        if n_pri < center_pri:
            masked_directions.add(d)

    print(f"center_pri={center_pri}")
    for d in NEIGHBOR_DIRECTIONS:
        if d in neighbor_priorities:
            p = neighbor_priorities[d]
            status = 'MASKED' if d in masked_directions else 'skip'
            print(f"  {d}: pri={p} {status}")

    # Build hex masks for center + neighbors
    # tile 0 = center, tiles 1-6 = neighbors by index
    dir_to_idx = {name: idx for idx, name in enumerate(NEIGHBOR_DIRECTIONS)}

    all_masks = []  # [center_mask, neighbor_mask_0, ..., neighbor_mask_5]
    all_priorities = [center_pri]  # priority for each tile

    # Center hex mask
    center_mask = _build_hex_mask_at(W, H, cx, cy, hex_radius)
    all_masks.append(center_mask)

    # Neighbor hex masks at offset positions
    neighbor_mask_by_dir = {}
    for d in NEIGHBOR_DIRECTIONS:
        q, r = _AXIAL_OFFSETS[d]
        ox, oy = _neighbor_pixel_offset(q, r, hex_radius)
        nmask = _build_hex_mask_at(W, H, cx + ox, cy + oy, hex_radius)
        idx = dir_to_idx[d] + 1  # offset by 1 (0 = center)
        all_masks.append(nmask)
        if d in neighbor_priorities:
            all_priorities.append(neighbor_priorities[d])
        else:
            all_priorities.append(None)
        if d in active_directions:
            neighbor_mask_by_dir[d] = nmask

    # Build tile_map: priority-ordered assignment
    # Highest priority (lowest number) first so overlap pixels go to preserved tile
    tile_map = torch.full((H, W), -1, dtype=torch.long)
    tile_order = list(range(len(all_masks)))
    # Filter to tiles with known priorities
    tiles_with_pri = [(i, all_priorities[i]) for i in tile_order if all_priorities[i] is not None]
    tiles_with_pri.sort(key=lambda x: x[1])  # highest priority (lowest number) first
    for tile_idx, _ in tiles_with_pri:
        unassigned = tile_map < 0
        tile_map[all_masks[tile_idx] & unassigned] = tile_idx

    # Angular fallback for uncovered pixels
    uncovered = tile_map < 0
    if uncovered.any():
        sectors = _compute_sector_map(W, H)
        tile_map[uncovered] = sectors[uncovered] + 1  # +1 because 0=center

    # Build eroded center mask for border ring
    erosion = max(1, int(border_width * hex_radius))
    pad = erosion + 1
    padded = F.pad(center_mask.float().unsqueeze(0).unsqueeze(0), [pad]*4, mode='constant', value=1.0)
    eroded_center = _erode_mask(padded.squeeze().bool(), erosion)
    if pad > 0:
        eroded_center = eroded_center[pad:-pad, pad:-pad]
    border_ring = center_mask & ~eroded_center

    # Build noise_mask using tile_map
    # For each masked direction: mask pixels in border ring that tile_map assigns to center
    # (because neighbor has higher priority, center pixels at the border need regeneration)
    feather = max(0, round(0.3 * erosion))
    noise_mask = torch.zeros(H, W, dtype=torch.float32)

    if feather > 0:
        dist_to_eroded = torch.from_numpy(
            np.clip(_manhattan_distance_to_region(eroded_center) / feather, 0.0, 1.0)
        )
    else:
        dist_to_eroded = torch.ones(H, W, dtype=torch.float32)

    sectors = _compute_sector_map(W, H)

    # Debug: check mask components
    print(f"\nMask debug:")
    print(f"  border_ring pixels: {border_ring.sum().item()}")
    print(f"  eroded_center pixels: {eroded_center.sum().item()}")
    print(f"  tile_map==0 in border: {(border_ring & (tile_map == 0)).sum().item()}")
    for d in sorted(masked_directions):
        di = dir_to_idx[d]
        in_sector = (sectors == di)
        in_border_sector = border_ring & in_sector
        owns_center = tile_map == 0
        all_filters = border_ring & owns_center & in_sector
        print(f"  {d} (sector {di}): sector_pixels={in_sector.sum().item()}, "
              f"border_sector={in_border_sector.sum().item()}, "
              f"owns_center_sector={all_filters.sum().item()}")

    for d in masked_directions:
        di = dir_to_idx[d]
        dir_idx = di + 1  # tile index for this neighbor
        # Only mask pixels assigned to center (tile 0) in the tile_map,
        # AND in this direction's sector. Overlap pixels assigned to the
        # higher-priority neighbor are excluded from the mask.
        owns_center = tile_map == 0
        dir_border = border_ring & owns_center & (sectors == di)
        if dir_border.any():
            noise_mask = torch.max(noise_mask, dir_border.float() * dist_to_eroded)

    # --- Three-step visualization ---
    # Simulate priority-ordered paste
    # Paste order: lowest priority (highest number) first, highest priority last
    paste_order = list(range(len(all_masks)))
    paste_with_pri = [(i, all_priorities[i]) for i in paste_order if all_priorities[i] is not None]
    paste_with_pri.sort(key=lambda x: -x[1])  # lowest priority first

    print(f"\nPaste order (lowest pri first):")
    for step, (tile_idx, pri) in enumerate(paste_with_pri):
        name = "center" if tile_idx == 0 else NEIGHBOR_DIRECTIONS[tile_idx - 1]
        print(f"  Step {step}: {name} (pri={pri})")

    owner = torch.full((H, W), -1, dtype=torch.long)
    overwrite_step = torch.zeros(H, W, dtype=torch.long)

    for step_num, (tile_idx, pri) in enumerate(paste_with_pri):
        mask = all_masks[tile_idx]
        fresh = mask & (owner < 0)
        overwrite = mask & (owner >= 0)

        owner[fresh] = tile_idx
        overwrite_step[fresh] = 0

        if overwrite.any():
            owner[overwrite] = tile_idx
            overwrite_step[overwrite] = step_num + 1

    # Build ASCII
    # Letters: center=a, first neighbor paste=b, second=c, etc.
    step_letter = {}
    for step_num, (tile_idx, pri) in enumerate(paste_with_pri):
        if tile_idx == 0:
            step_letter[tile_idx] = 'a'
        else:
            step_letter[tile_idx] = chr(ord('b') + step_num - 1)

    ascii_grid = []
    for y in range(H):
        row = []
        for x in range(W):
            if not center_mask[y, x] and owner[y, x] < 0:
                row.append('.')
                continue

            ow = overwrite_step[y, x].item()
            is_masked = noise_mask[y, x].item() > 0
            t_idx = owner[y, x].item()

            if ow == 0:
                base = step_letter.get(t_idx, '?')
            elif ow == 1:
                base = 'x'
            else:
                base = 'y'

            if is_masked:
                row.append(base.upper())
            else:
                row.append(base.lower())
        ascii_grid.append(row)

    # Counts
    counts = {}
    for y in range(H):
        for x in range(W):
            c = ascii_grid[y][x]
            counts[c] = counts.get(c, 0) + 1

    # Debug: where are the Y pixels?
    Y_pixels = []
    for y in range(H):
        for x in range(W):
            if ascii_grid[y][x] == 'Y':
                t_idx = owner[y, x].item()
                ow = overwrite_step[y, x].item()
                sec = _compute_sector_map(W, H)[y, x].item() if y < H and x < W else -1
                Y_pixels.append((y, x, t_idx, ow, sec, NEIGHBOR_DIRECTIONS[sec] if 0 <= sec < 6 else '?'))

    print(f"\nPixel counts:")
    for c in sorted(counts.keys()):
        print(f"  {c}: {counts[c]}")

    if Y_pixels:
        print(f"\nY pixels breakdown (owner, overwrite_step, sector):")
        from collections import Counter
        owner_counts = Counter(t[2] for t in Y_pixels)
        sec_counts = Counter(t[4] for t in Y_pixels)
        print(f"  By owner tile: {dict(owner_counts)}")
        print(f"  By sector: {dict(sec_counts)}")
        # Show a few examples
        for y, x, t, ow, sec, sname in Y_pixels[:10]:
            name = "center" if t == 0 else NEIGHBOR_DIRECTIONS[t-1] if 0 < t <= 6 else f"tile{t}"
            print(f"    ({y},{x}): owner={name}, ow_step={ow}, sector={sname}")

    # Find border bounds
    border_ys, border_xs = torch.where(border_ring & center_mask)
    if len(border_ys) > 0:
        min_y = max(0, border_ys.min().item() - 3)
        max_y = min(H, border_ys.max().item() + 4)
        min_x = max(0, border_xs.min().item() - 3)
        max_x = min(W, border_xs.max().item() + 4)

    # NE border closeup
    print(f"\nNE border closeup (rows {min_y}-{min_y+30}, cols {min_x}-{min_x+50}):")
    for y in range(min_y, min(min_y + 30, H)):
        row_str = ''.join(ascii_grid[y][min_x:min(min_x + 50, W)])
        print(f"{y:3d} |{row_str}|")

    # E border closeup
    e_min_y = max(0, H // 2 - 15)
    e_min_x = max(0, W - 50)
    print(f"\nE border closeup (rows {e_min_y}-{e_min_y+30}, cols {e_min_x}-{W}):")
    for y in range(e_min_y, min(e_min_y + 30, H)):
        row_str = ''.join(ascii_grid[y][e_min_x:W])
        print(f"{y:3d} |{row_str}|")

    # Corner between E and NE (where shared edge should be)
    # This is around the vertex at the top-right of the hex
    print(f"\nE-NE corner closeup (rows 20-50, cols 110-150):")
    for y in range(20, min(50, H)):
        row_str = ''.join(ascii_grid[y][110:min(150, W)])
        print(f"{y:3d} |{row_str}|")


if __name__ == "__main__":
    visualize()
