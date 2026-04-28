"""Tests for corner composition and mask generation (pure PyTorch)."""

import math
import torch
import sys
import os
import importlib

# Set up package imports so relative imports work within the package
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_pkg_dir)
sys.path.insert(0, _parent)

# The directory has a hyphen, so use importlib for the top-level package
_pkg_name = os.path.basename(_pkg_dir)
_pkg = importlib.import_module(_pkg_name)
sys.modules[_pkg_name] = _pkg
# Also register it without hyphen for relative imports
sys.modules["ComfyUI_AdvancedTiling"] = _pkg

from corner_composite import (
    CORNER_NAMES,
    CORNER_NEIGHBORS,
    get_corner_offsets,
    build_corner_tile_map,
    composite_corner_preview,
    _build_offset_hex_mask,
)
from modes.hex_mask import (
    create_corner_masks, _build_hex_mask_at, NEIGHBOR_DIRECTIONS,
)


def test_sector_map_at_center_matches_global():
    """_compute_sector_map_at centered on image center should match _compute_sector_map."""
    from modes.hex_mask import _compute_sector_map, _compute_sector_map_at
    W, H = 256, 256
    global_map = _compute_sector_map(W, H)
    offset_map = _compute_sector_map_at(W, H, W / 2.0, H / 2.0)
    assert (global_map == offset_map).all(), "Sector maps should match at image center"


def test_sector_map_at_offset_covers_all_sectors():
    """Sector map at offset position should still produce sectors 0-5."""
    from modes.hex_mask import _compute_sector_map_at
    W, H = 256, 256
    smap = _compute_sector_map_at(W, H, 100.0, 150.0)
    unique = smap.unique().tolist()
    assert set(unique) == {0, 1, 2, 3, 4, 5}, f"Expected all 6 sectors, got {unique}"


def test_corner_edge_sectors_structure():
    """Each corner should have 3 edges, each with 2 (tile, sector) pairs."""
    from modes.hex_mask import _CORNER_EDGE_SECTORS
    for corner, edges in _CORNER_EDGE_SECTORS.items():
        assert len(edges) == 3, f"{corner}: expected 3 edges, got {len(edges)}"
        for tile_a, sec_a, tile_b, sec_b in edges:
            assert tile_a in {0, 1, 2}, f"{corner}: invalid tile_a={tile_a}"
            assert tile_b in {0, 1, 2}, f"{corner}: invalid tile_b={tile_b}"
            assert 0 <= sec_a <= 5, f"{corner}: invalid sector_a={sec_a}"
            assert 0 <= sec_b <= 5, f"{corner}: invalid sector_b={sec_b}"
            assert tile_a != tile_b, f"{corner}: same tile on both sides"


def test_corner_mask_sector_shape_matches_central():
    """Corner mask sectors should have same shape as CentralTile neighbor masks."""
    from modes.hex_mask import create_corner_masks, _build_hex_mask_at, _compute_sector_map_at
    W, H = 256, 256
    R = W // 2
    corner = "NE"

    corner_mask = create_corner_masks(W, H, corner, border_width=0.2, tile_priorities=None)
    assert corner_mask.max() > 0, "Corner mask should be non-empty with no priorities"
    assert corner_mask[0, 0, 0] == 0, "Mask should be zero at top-left corner"
    assert corner_mask[0, 0, -1] == 0, "Mask should be zero at top-right corner"
    assert corner_mask[0, -1, 0] == 0, "Mask should be zero at bottom-left corner"
    assert corner_mask[0, -1, -1] == 0, "Mask should be zero at bottom-right corner"


def test_corner_mask_half_erosion_unknown_priority():
    """When both priorities are unknown, each side uses half border width."""
    from modes.hex_mask import create_corner_masks
    W, H = 256, 256

    mask_unknown = create_corner_masks(W, H, "NE", border_width=0.2, tile_priorities=None)
    mask_known = create_corner_masks(W, H, "NE", border_width=0.2, tile_priorities=[1, 2, 3])

    assert mask_unknown.max() > 0, "Unknown priorities should produce mask"
    assert mask_known.max() > 0, "Known different priorities should produce mask"


def test_corner_offsets_symmetry():
    """All corners should have symmetric offsets (same magnitude, rotated)."""
    R = 256
    for corner in CORNER_NAMES:
        offsets = get_corner_offsets(corner, R)
        for ox, oy in offsets:
            dist = math.sqrt(ox * ox + oy * oy)
            assert abs(dist - R) < 2.0, (
                f"{corner}: offset ({ox},{oy}) distance {dist:.1f} != R={R}"
            )
    print("PASS: test_corner_offsets_symmetry")


def test_tile_map_covers_all():
    """Tile map should assign every pixel to one of 3 tiles."""
    for corner in CORNER_NAMES:
        tile_map = build_corner_tile_map(256, 256, corner, 128)
        unique = tile_map.unique()
        assert set(unique.tolist()).issubset({0, 1, 2}), (
            f"{corner}: unexpected tile indices {unique.tolist()}"
        )
        assert tile_map.numel() == 256 * 256
    print("PASS: test_tile_map_covers_all")


def test_corner_preview_shape():
    """Corner preview should be square (cropped)."""
    B, H, W, C = 1, 256, 256, 3
    img_a = torch.rand(B, H, W, C)
    img_b = torch.rand(B, H, W, C)
    img_c = torch.rand(B, H, W, C)

    for corner in CORNER_NAMES:
        preview = composite_corner_preview(img_a, img_b, img_c, corner)
        _, pH, pW, _ = preview.shape
        assert pH == pW, f"{corner}: not square: {pH}x{pW}"
        assert pH <= H, f"{corner}: crop height {pH} > input {H}"
    print("PASS: test_corner_preview_shape")


def test_corner_mask_shape_and_range():
    """Corner mask should be (1, H, W) with values in [0, 1]."""
    for corner in CORNER_NAMES:
        mask = create_corner_masks(256, 256, corner, border_width=0.2)
        assert mask.shape == (1, 256, 256), f"{corner}: shape {mask.shape}"
        assert mask.min() >= 0.0, f"{corner}: min {mask.min()}"
        assert mask.max() <= 1.0, f"{corner}: max {mask.max()}"
    print("PASS: test_corner_mask_shape_and_range")


def test_corner_mask_equal_priority_no_mask():
    """When all 3 tiles have equal priority, mask should be empty."""
    for corner in CORNER_NAMES:
        mask = create_corner_masks(
            256, 256, corner, border_width=0.2,
            tile_priorities=[3, 3, 3],
        )
        assert mask.max() == 0.0, (
            f"{corner}: equal priority should produce empty mask, got max={mask.max()}"
        )
    print("PASS: test_corner_mask_equal_priority_no_mask")


def test_corner_mask_extent():
    """half_edge should produce smaller mask than full_edge."""
    corner = "NE"
    mask_half = create_corner_masks(256, 256, corner, border_width=0.2, mask_extent="half_edge")
    mask_full = create_corner_masks(256, 256, corner, border_width=0.2, mask_extent="full_edge")
    assert mask_half.sum() < mask_full.sum(), (
        f"half_edge ({mask_half.sum()}) should be < full_edge ({mask_full.sum()})"
    )
    print("PASS: test_corner_mask_extent")


def test_corner_neighbors_mapping():
    """Each corner should map to valid neighbor directions."""
    for corner, (n1, n2) in CORNER_NEIGHBORS.items():
        assert n1 in NEIGHBOR_DIRECTIONS, f"{corner}: n1={n1} not in directions"
        assert n2 in NEIGHBOR_DIRECTIONS, f"{corner}: n2={n2} not in directions"
        assert n1 != n2, f"{corner}: n1 and n2 should be different"
    print("PASS: test_corner_neighbors_mapping")


def test_hex_mask_covers_center():
    """Hex mask centered at image center should cover the center pixel."""
    mask = _build_hex_mask_at(256, 256, 128, 128, 128)
    assert mask[128, 128], "Center pixel should be inside hex"
    print("PASS: test_hex_mask_covers_center")


def test_hex_mask_offset_covers_vertex():
    """Offset hex mask should cover the output center (vertex position)."""
    R = 128
    # For corner N, central tile offset is (0, R)
    mask = _build_offset_hex_mask(256, 256, 0, R, R)
    assert mask[128, 128], "Vertex at output center should be inside offset hex"
    print("PASS: test_hex_mask_offset_covers_vertex")


def test_tile_map_center_assigned_to_central():
    """Center pixel should be assigned to the central tile (index 0)."""
    for corner in CORNER_NAMES:
        tile_map = build_corner_tile_map(256, 256, corner, 128)
        assert tile_map[128, 128] == 0, (
            f"{corner}: center pixel assigned to tile {tile_map[128, 128]}, expected 0"
        )
    print("PASS: test_tile_map_center_assigned_to_central")


def test_no_black_pixels_in_preview():
    """Cropped preview should have no black pixels (all content)."""
    B, H, W, C = 1, 256, 256, 3
    img_a = torch.ones(B, H, W, C) * 0.5
    img_b = torch.ones(B, H, W, C) * 0.7
    img_c = torch.ones(B, H, W, C) * 0.9

    for corner in CORNER_NAMES:
        preview = composite_corner_preview(img_a, img_b, img_c, corner)
        black_pixels = (preview.sum(dim=-1) == 0).sum().item()
        assert black_pixels == 0, (
            f"{corner}: {black_pixels} black pixels in cropped preview"
        )
    print("PASS: test_no_black_pixels_in_preview")


if __name__ == "__main__":
    test_sector_map_at_center_matches_global()
    test_sector_map_at_offset_covers_all_sectors()
    test_corner_edge_sectors_structure()
    test_corner_mask_sector_shape_matches_central()
    test_corner_mask_half_erosion_unknown_priority()
    test_corner_offsets_symmetry()
    test_tile_map_covers_all()
    test_corner_preview_shape()
    test_corner_mask_shape_and_range()
    test_corner_mask_equal_priority_no_mask()
    test_corner_mask_extent()
    test_corner_neighbors_mapping()
    test_hex_mask_covers_center()
    test_hex_mask_offset_covers_vertex()
    test_tile_map_center_assigned_to_central()
    test_no_black_pixels_in_preview()
    print("\nAll tests passed!")
