"""
CornerEdges NE corner ASCII visualization.
Three-step priority-ordered pasting:
  Step 1: paste lowest-priority tile  -> a/A (tile content)
  Step 2: paste mid-priority tile     -> b/B (new), x/X (overwrites step 1)
  Step 3: paste highest-priority tile -> c/C (new), y/Y (overwrites step 1 or 2)
  uppercase = masked (regenerate), lowercase = not masked (preserved)
  . = outside all hex masks
"""

import sys
import os
import math
import torch
import numpy as np

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_pkg_dir)
sys.path.insert(0, _parent)

import importlib
_pkg_name = os.path.basename(_pkg_dir)
_pkg = importlib.import_module(_pkg_name)
sys.modules[_pkg_name] = _pkg
sys.modules["ComfyUI_AdvancedTiling"] = _pkg

from corner_composite import (
    CORNER_NAMES, _CORNER_OFFSETS, _build_offset_hex_mask,
    build_corner_tile_map, get_corner_offsets,
)
from modes.hex_mask import create_corner_masks


def visualize_corner():
    corner = "NE"
    W, H = 128, 128
    hex_radius = W // 2  # 64
    border_width = 0.2

    # Tile priorities: [center=5, n1=3, n2=1]
    # Priority order for pasting: tile 0 (pri=5) first, tile 1 (pri=3), tile 2 (pri=1) last
    tile_priorities = [5, 3, 1]  # center=lowest priority, n2=highest priority

    # The composite pastes in order: [tile_0(pri=5), tile_1(pri=3), tile_2(pri=1)]
    # (lowest priority = highest number = pasted first)
    paste_order = list(range(3))
    paste_order.sort(key=lambda i: -tile_priorities[i])  # [0, 1, 2]

    print(f"Corner: {corner}")
    print(f"Tile priorities: center={tile_priorities[0]}, n1={tile_priorities[1]}, n2={tile_priorities[2]}")
    print(f"Paste order: {paste_order} (lowest priority first, highest last)")
    print(f"  Step 1: tile {paste_order[0]} (pri={tile_priorities[paste_order[0]]}) -> a/A")
    print(f"  Step 2: tile {paste_order[1]} (pri={tile_priorities[paste_order[1]]}) -> b/B new, x/X overwrite")
    print(f"  Step 3: tile {paste_order[2]} (pri={tile_priorities[paste_order[2]]}) -> c/C new, y/Y overwrite")

    # Build hex masks
    unit_offsets = _CORNER_OFFSETS[corner]
    float_offsets = [(ux * hex_radius, uy * hex_radius) for ux, uy in unit_offsets]
    int_offsets = get_corner_offsets(corner, hex_radius)

    hex_masks = []
    for ox, oy in float_offsets:
        hm = _build_offset_hex_mask(W, H, ox, oy, hex_radius)
        hex_masks.append(hm)

    # Build tile_map (priority-ordered: highest priority first)
    tile_map = torch.full((H, W), -1, dtype=torch.long)
    tile_order_map = list(range(3))
    tile_order_map.sort(key=lambda i: tile_priorities[i])  # highest priority first
    for tile_idx in tile_order_map:
        unassigned = tile_map < 0
        tile_map[hex_masks[tile_idx] & unassigned] = tile_idx

    # Angular fallback
    uncovered = tile_map < 0
    if uncovered.any():
        fb_map = build_corner_tile_map(W, H, corner, hex_radius)
        tile_map[uncovered] = fb_map[uncovered]

    # Generate corner mask (noise_mask: 1=regenerate, 0=preserve)
    feather = max(0, round(0.3 * max(1, int(border_width * hex_radius))))
    noise_mask = create_corner_masks(
        W, H, corner, border_width, feather, mask_extent="half_edge",
        tile_priorities=tile_priorities,
    )
    # noise_mask is (1, H, W), squeeze to (H, W)
    noise_mask_2d = noise_mask.squeeze(0)

    # --- Three-step simulation ---
    # Track what each pixel was BEFORE each step and AFTER
    owner = torch.full((H, W), -1, dtype=torch.long)  # -1=unassigned
    overwrite_step = torch.zeros(H, W, dtype=torch.long)
    # overwrite_step: 0=original (tile letter), 1=overwritten by step 2 (x), 2=overwritten by step 3 (y)

    for step_idx, tile_idx in enumerate(paste_order):
        mask = hex_masks[tile_idx]
        # Also include angular fallback pixels for this tile
        fb_pixels = uncovered & (tile_map == tile_idx) if uncovered.any() else torch.zeros(H, W, dtype=torch.bool)
        paste_region = mask | fb_pixels

        newly_overwriting = paste_region & (owner >= 0)  # was already assigned
        fresh = paste_region & (owner < 0)  # wasn't assigned yet

        owner[fresh] = tile_idx
        overwrite_step[fresh] = 0  # fresh tile content

        owner[newly_overwriting] = tile_idx
        if step_idx == 1:
            overwrite_step[newly_overwriting] = 1  # x/X: overwritten by step 2
        elif step_idx == 2:
            overwrite_step[newly_overwriting] = 2  # y/Y: overwritten by step 3

    # Build ASCII
    # tile_idx -> letter base: step1_tile->a, step2_tile->b, step3_tile->c
    step_tile = {paste_order[0]: 'a', paste_order[1]: 'b', paste_order[2]: 'c'}

    ascii_grid = []
    for y in range(H):
        row = []
        for x in range(W):
            if owner[y, x] < 0:
                row.append('.')
                continue

            is_masked = noise_mask_2d[y, x].item() > 0
            ow = overwrite_step[y, x].item()
            t_idx = owner[y, x].item()

            if ow == 0:
                # Fresh tile content
                base = step_tile[t_idx]
            elif ow == 1:
                base = 'x'  # overwritten by step 2
            elif ow == 2:
                base = 'y'  # overwritten by step 3
            else:
                base = '?'

            # uppercase = masked, lowercase = preserved
            if is_masked:
                row.append(base.upper())
            else:
                row.append(base.lower())
        ascii_grid.append(row)

    # Counts
    total_a = sum(1 for y in range(H) for x in range(W) if ascii_grid[y][x] in ('a', 'A'))
    total_b = sum(1 for y in range(H) for x in range(W) if ascii_grid[y][x] in ('b', 'B'))
    total_c = sum(1 for y in range(H) for x in range(W) if ascii_grid[y][x] in ('c', 'C'))
    total_x = sum(1 for y in range(H) for x in range(W) if ascii_grid[y][x] in ('x', 'X'))
    total_y = sum(1 for y in range(H) for x in range(W) if ascii_grid[y][x] in ('y', 'Y'))
    total_dot = sum(1 for y in range(H) for x in range(W) if ascii_grid[y][x] == '.')

    total_masked = sum(1 for y in range(H) for x in range(W) if ascii_grid[y][x].isupper())

    print(f"\nPixel counts:")
    print(f"  a (tile {paste_order[0]}): {total_a}")
    print(f"  b (tile {paste_order[1]}): {total_b}")
    print(f"  c (tile {paste_order[2]}): {total_c}")
    print(f"  x (overwritten by step 2): {total_x}")
    print(f"  y (overwritten by step 3): {total_y}")
    print(f"  . (outside): {total_dot}")
    print(f"  Total masked (uppercase): {total_masked}")

    # Find the vertex area (center of output) for closeup
    cx, cy = W // 2, H // 2

    # Print closeup around the vertex (where 3 tiles meet)
    r = 25
    print(f"\nCloseup around vertex (rows {cy-r}-{cy+r-1}, cols {cx-r}-{cx+r-1}):")
    print(f"  a/b/c = tile content (preserved), A/B/C = tile content (masked)")
    print(f"  x = overwritten by step 2 (preserved), X = overwritten by step 2 (masked)")
    print(f"  y = overwritten by step 3 (preserved), Y = overwritten by step 3 (masked)")
    print(f"  . = outside all hex masks")
    print()

    for y in range(max(0, cy - r), min(H, cy + r)):
        row_str = ''.join(ascii_grid[y][max(0, cx - r):min(W, cx + r)])
        print(f"{y:3d} |{row_str}|")


if __name__ == "__main__":
    visualize_corner()
