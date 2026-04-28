"""Tests for corner composition and mask generation (pure PyTorch)."""

import math
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from corner_composite import (
    CORNER_NAMES,
    CORNER_NEIGHBORS,
    get_corner_offsets,
    build_corner_tile_map,
    composite_corner_preview,
)
from modes.hex_mask import create_corner_masks


def test_corner_offsets_symmetry():
    """All corners should have symmetric offsets (same magnitude, rotated)."""
    R = 256
    for corner in CORNER_NAMES:
        offsets = get_corner_offsets(corner, R)
        # Each offset should have magnitude R (vertex distance from center)
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
        # All pixels assigned
        assert tile_map.numel() == 256 * 256
    print("PASS: test_tile_map_covers_all")


def test_corner_preview_shape():
    """Corner preview should have same shape as input."""
    B, H, W, C = 1, 256, 256, 3
    img_a = torch.rand(B, H, W, C)
    img_b = torch.rand(B, H, W, C)
    img_c = torch.rand(B, H, W, C)

    for corner in CORNER_NAMES:
        preview = composite_corner_preview(img_a, img_b, img_c, corner)
        assert preview.shape == (B, H, W, C), (
            f"{corner}: shape {preview.shape} != {(B, H, W, C)}"
        )
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
            tile_priorities=[3, 3, 3],  # all equal
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
    from modes.hex_mask import NEIGHBOR_DIRECTIONS
    for corner, (n1, n2) in CORNER_NEIGHBORS.items():
        assert n1 in NEIGHBOR_DIRECTIONS, f"{corner}: n1={n1} not in directions"
        assert n2 in NEIGHBOR_DIRECTIONS, f"{corner}: n2={n2} not in directions"
        assert n1 != n2, f"{corner}: n1 and n2 should be different"
    print("PASS: test_corner_neighbors_mapping")


if __name__ == "__main__":
    test_corner_offsets_symmetry()
    test_tile_map_covers_all()
    test_corner_preview_shape()
    test_corner_mask_shape_and_range()
    test_corner_mask_equal_priority_no_mask()
    test_corner_mask_extent()
    test_corner_neighbors_mapping()
    print("\nAll tests passed!")
