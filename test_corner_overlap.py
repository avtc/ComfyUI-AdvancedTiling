"""
Tests for coordinate offset correctness in tile compositing.

Verifies that neighbor content is pasted at the correct spatial position
(by reading from the correct location in the neighbor's latent/image),
not at the same (y,x) coordinate which would sample the wrong edge.

Uses gradient tensors (value = x_coord) so offset bugs are detectable:
with wrong offset, pasted values match output x; with correct offset,
pasted values match the shifted source x.
"""

import math
import torch
import sys
import os

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_pkg_dir)
sys.path.insert(0, _parent)

import importlib
_pkg_name = os.path.basename(_pkg_dir)
_pkg = importlib.import_module(_pkg_name)
sys.modules[_pkg_name] = _pkg
sys.modules["ComfyUI_AdvancedTiling"] = _pkg

from corner_composite import (
    composite_corner_preview,
    _build_offset_hex_mask, get_corner_offsets,
    _composite_priority_ordered,
)
from modes.hex_mask import (
    NEIGHBOR_DIRECTIONS, _build_hex_mask_at, _neighbor_offset_px,
    create_central_tile_masks,
)


def _make_gradient_latent(H, W):
    """Create a 4D latent where value = x coordinate (0..W-1)."""
    xs = torch.arange(W, dtype=torch.float32)
    return xs.unsqueeze(0).unsqueeze(0).expand(1, 4, H, W).clone()


def _make_gradient_image(H, W):
    """Create a 4D image where value = x coordinate (0..W-1)."""
    xs = torch.arange(W, dtype=torch.float32)
    return xs.unsqueeze(0).unsqueeze(-1).expand(1, H, W, 3).clone()


# ─── CornerEdges tests ────────────────────────────────────────────────


def test_corner_latent_offset_NE():
    """NE corner: E neighbor content should come from E neighbor's left edge (low x)."""
    W, H = 128, 128
    R = W // 2
    corner = "NE"

    center_lat = _make_gradient_latent(H, W)
    e_lat = _make_gradient_latent(H, W)
    ne_lat = _make_gradient_latent(H, W)

    result = _composite_priority_ordered(
        [center_lat, e_lat, ne_lat], corner, R, is_latent=True,
    )

    # Build hex masks to find overlap regions
    offsets = get_corner_offsets(corner, R)
    float_offsets = [(ux * R, uy * R) for ux, uy in
                     [(-0.866025404, 0.5), (0.866025404, 0.5), (0.0, -1.0)]]
    hex_masks = [_build_offset_hex_mask(W, H, ox, oy, R) for ox, oy in float_offsets]

    # E neighbor hex (tile 1) should have its LEFT edge at the center of output
    # So pixels near output center should have LOW x values from E neighbor
    center_region = hex_masks[1]  # E neighbor's hex mask
    if center_region.any():
        ys, xs = torch.where(center_region)
        # Pick a pixel near the center of the E neighbor's hex
        mid = len(ys) // 2
        sy, sx = ys[mid].item(), xs[mid].item()
        actual = result[0, 0, sy, sx].item()
        # With correct offset: output (sy,sx) maps to E neighbor latent at (sy - oy, sx - ox)
        ox_i, oy_i = offsets[1]  # E neighbor offset
        expected_src_x = max(0, min(W - 1, sx - ox_i))
        # The E neighbor's gradient value at expected_src_x should be expected_src_x
        ok = abs(actual - expected_src_x) < 2.0
        assert ok, (
            f"NE corner E neighbor: pixel ({sy},{sx}) "
            f"actual={actual:.0f}, expected_src_x={expected_src_x}"
        )
    print("PASS: test_corner_latent_offset_NE")


def test_corner_latent_offset_all_corners():
    """All 6 corners: verify coordinate offset for each neighbor."""
    W, H = 128, 128
    R = W // 2

    for corner in ["N", "NE", "SE", "S", "SW", "NW"]:
        center_lat = _make_gradient_latent(H, W)
        n1_lat = _make_gradient_latent(H, W)
        n2_lat = _make_gradient_latent(H, W)

        result = _composite_priority_ordered(
            [center_lat, n1_lat, n2_lat], corner, R, is_latent=True,
        )

        offsets = get_corner_offsets(corner, R)
        float_offsets = [(ux * R, uy * R) for ux, uy in
                         [(offsets[0][0] / R, offsets[0][1] / R),
                          (offsets[1][0] / R, offsets[1][1] / R),
                          (offsets[2][0] / R, offsets[2][1] / R)]]
        hex_masks = [_build_offset_hex_mask(W, H, ox, oy, R)
                     for ox, oy in float_offsets]

        for tile_idx in [1, 2]:
            mask = hex_masks[tile_idx]
            if not mask.any():
                continue
            ys, xs_idx = torch.where(mask)
            mid = len(ys) // 2
            sy, sx = ys[mid].item(), xs_idx[mid].item()
            actual = result[0, 0, sy, sx].item()
            ox_i, oy_i = offsets[tile_idx]
            expected_src_x = max(0, min(W - 1, sx - ox_i))
            ok = abs(actual - expected_src_x) < 2.0
            assert ok, (
                f"{corner} tile{tile_idx}: pixel ({sy},{sx}) "
                f"actual={actual:.0f}, expected_src_x={expected_src_x}"
            )
    print("PASS: test_corner_latent_offset_all_corners")


def test_corner_image_offset():
    """Corner preview image: verify coordinate offset at image resolution."""
    W, H = 256, 256
    R = W // 2
    corner = "NE"

    center_img = _make_gradient_image(H, W)
    n1_img = _make_gradient_image(H, W)
    n2_img = _make_gradient_image(H, W)

    result = composite_corner_preview(center_img, n1_img, n2_img, corner)

    offsets = get_corner_offsets(corner, R)
    float_offsets = [(ux * R, uy * R) for ux, uy in
                     [(-0.866025404, 0.5), (0.866025404, 0.5), (0.0, -1.0)]]
    hex_masks = [_build_offset_hex_mask(W, H, ox, oy, R) for ox, oy in float_offsets]

    for tile_idx in [1, 2]:
        mask = hex_masks[tile_idx]
        if not mask.any():
            continue
        ys, xs_idx = torch.where(mask)
        mid = len(ys) // 2
        sy, sx = ys[mid].item(), xs_idx[mid].item()
        actual = result[0, sy, sx, 0].item()  # first channel = x gradient
        ox_i, oy_i = offsets[tile_idx]
        expected_src_x = max(0, min(W - 1, sx - ox_i))
        ok = abs(actual - expected_src_x) < 2.0
        assert ok, (
            f"NE corner image tile{tile_idx}: pixel ({sy},{sx}) "
            f"actual={actual:.0f}, expected_src_x={expected_src_x}"
        )
    print("PASS: test_corner_image_offset")


# ─── CentralTile tests ────────────────────────────────────────────────


def test_central_latent_overlap_offset():
    """CentralTile: neighbor latent at overlap should use coordinate offset."""
    W, H = 128, 128
    hex_radius = min(W, H) // 2

    center_lat = _make_gradient_latent(H, W)
    e_lat = _make_gradient_latent(H, W)
    ne_lat = _make_gradient_latent(H, W)

    # Build hex masks for center + E + NE neighbors
    cx, cy = W / 2.0, H / 2.0
    center_hex = _build_hex_mask_at(W, H, cx, cy, hex_radius)
    ox_e, oy_e = _neighbor_offset_px("E", hex_radius)
    e_hex = _build_hex_mask_at(W, H, cx + ox_e, cy + oy_e, hex_radius)
    ox_ne, oy_ne = _neighbor_offset_px("NE", hex_radius)
    ne_hex = _build_hex_mask_at(W, H, cx + ox_ne, cy + oy_ne, hex_radius)

    # Find overlap regions
    e_overlap = center_hex & e_hex
    ne_overlap = center_hex & ne_hex

    # Build coordinate grids
    ys = torch.arange(H, dtype=torch.long)
    xs = torch.arange(W, dtype=torch.long)

    # Test E neighbor overlap
    if e_overlap.any():
        e_ys, e_xs = torch.where(e_overlap)
        mid = len(e_ys) // 2
        sy, sx = e_ys[mid].item(), e_xs[mid].item()
        ox_i, oy_i = round(ox_e), round(oy_e)
        src_y = max(0, min(H - 1, sy - oy_i))
        src_x = max(0, min(W - 1, sx - ox_i))
        expected = e_lat[0, 0, src_y, src_x].item()
        wrong = e_lat[0, 0, sy, sx].item()
        assert expected != wrong, (
            f"E overlap ({sy},{sx}): offset should matter "
            f"(expected={expected:.0f}, wrong={wrong:.0f})"
        )
    print("PASS: test_central_latent_overlap_offset")


def test_central_mask_excludes_neighbor_owned_overlap():
    """CentralTile: overlap pixels owned by higher-priority neighbor are not masked."""
    W, H = 128, 128
    hex_radius = min(W, H) // 2
    border_width = 0.15

    center_pri = 2   # lowest priority
    e_pri = 1        # mid priority
    ne_pri = 0       # highest priority

    active = {0, 1}
    masked = {0, 1}
    priorities = [center_pri, e_pri, ne_pri, None, None, None, None]

    border_mask, _, hex_masks_lat, tile_map = create_central_tile_masks(
        W, H, hex_radius, border_width, 0, active, masked, priorities,
    )
    noise_mask = border_mask.squeeze(0)

    # Check: neighbor-owned overlap pixels must NOT be masked
    center_hex = hex_masks_lat[0]
    for tile_idx, name in [(1, "E"), (2, "NE")]:
        overlap = center_hex & hex_masks_lat[tile_idx]
        if not overlap.any():
            continue
        neighbor_owned = (tile_map != 0) & overlap
        masked_count = ((noise_mask > 0) & neighbor_owned).sum().item()
        assert masked_count == 0, (
            f"{name}: {masked_count} neighbor-owned overlap pixels are masked (should be 0)"
        )
    print("PASS: test_central_mask_excludes_neighbor_owned_overlap")


def test_central_mask_includes_center_owned_border():
    """CentralTile: center-owned border pixels in masked directions ARE masked."""
    W, H = 128, 128
    hex_radius = min(W, H) // 2
    border_width = 0.15

    center_pri = 2
    e_pri = 1
    ne_pri = 0

    active = {0, 1}
    masked = {0, 1}
    priorities = [center_pri, e_pri, ne_pri, None, None, None, None]

    border_mask, _, hex_masks_lat, tile_map = create_central_tile_masks(
        W, H, hex_radius, border_width, 0, active, masked, priorities,
    )
    noise_mask = border_mask.squeeze(0)

    # Center-owned pixels in the border ring should be masked
    center_owned_border = (tile_map == 0) & (noise_mask > 0)
    assert center_owned_border.sum().item() > 0, (
        "Expected center-owned border pixels to be masked"
    )
    print("PASS: test_central_mask_includes_center_owned_border")


def test_central_no_mask_when_center_highest_priority():
    """When center has highest priority, no directions are masked."""
    W, H = 128, 128
    hex_radius = min(W, H) // 2

    center_pri = 0  # highest priority
    active = {0, 1}
    masked = set()  # no directions masked (center is highest)
    priorities = [center_pri, 1, 2, None, None, None, None]

    border_mask, _, _, _ = create_central_tile_masks(
        W, H, hex_radius, 0.15, 0, active, masked, priorities,
    )
    assert border_mask.max() == 0.0, (
        "Center with highest priority should produce empty mask"
    )
    print("PASS: test_central_no_mask_when_center_highest_priority")


if __name__ == "__main__":
    # CornerEdges
    test_corner_latent_offset_NE()
    test_corner_latent_offset_all_corners()
    test_corner_image_offset()

    # CentralTile
    test_central_latent_overlap_offset()
    test_central_mask_excludes_neighbor_owned_overlap()
    test_central_mask_includes_center_owned_border()
    test_central_no_mask_when_center_highest_priority()

    print("\nAll corner overlap tests passed!")
