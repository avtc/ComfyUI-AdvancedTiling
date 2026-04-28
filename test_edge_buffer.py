"""Tests for edge buffer mask generation (pure PyTorch)."""

import sys
import os
import importlib
import torch

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_pkg_dir)
sys.path.insert(0, _parent)

_pkg_name = os.path.basename(_pkg_dir)
_pkg = importlib.import_module(_pkg_name)
sys.modules[_pkg_name] = _pkg
sys.modules["ComfyUI_AdvancedTiling"] = _pkg

from modes.hex_mask import (
    compute_edge_buffer_mask,
    create_feathered_masks,
    _build_inside_mask,
    NEIGHBOR_DIRECTIONS,
)
from modes import Settings


def _make_settings(tile_size=512):
    return Settings("Hexagon", 0, 1.0)


def test_buffer_depth_0_is_subset_of_border():
    """Buffer at depth=0 should be a subset of the border ring."""
    S = _make_settings()
    W, H = 64, 64
    active = {0, 1, 2, 3, 4, 5}

    inside, border_mask, _ = create_feathered_masks(W, H, S, 0.2, 0, True, active)
    buffer = compute_edge_buffer_mask(W, H, S, 0.2, 0, active)

    border_bool = border_mask.squeeze(0) > 0
    for d in active:
        # Every buffer pixel must be in the border ring
        assert not (buffer[d] & ~border_bool).any(), (
            f"Direction {d}: buffer pixels outside border ring"
        )
    print("PASS: test_buffer_depth_0_is_subset_of_border")


def test_buffer_depth_2_superset_of_depth_0():
    """Deeper buffer should contain all pixels from shallower buffer."""
    S = _make_settings()
    W, H = 64, 64
    active = {0, 1, 2, 3, 4, 5}

    buf0 = compute_edge_buffer_mask(W, H, S, 0.2, 0, active)
    buf2 = compute_edge_buffer_mask(W, H, S, 0.2, 2, active)

    for d in active:
        assert not (buf0[d] & ~buf2[d]).any(), (
            f"Direction {d}: depth=0 pixels missing from depth=2"
        )
    print("PASS: test_buffer_depth_2_superset_of_depth_0")


def test_buffer_only_active_directions():
    """Only active directions should have non-zero buffer masks."""
    S = _make_settings()
    W, H = 64, 64
    active = {0, 3}

    buffer = compute_edge_buffer_mask(W, H, S, 0.2, 0, active)

    for d in range(6):
        if d in active:
            # Active direction may or may not have pixels (depends on geometry)
            pass
        else:
            assert not buffer[d].any(), (
                f"Direction {d}: inactive but has buffer pixels"
            )
    print("PASS: test_buffer_only_active_directions")


def test_buffer_inside_hex_only():
    """Buffer pixels must be inside the hex (not in waste area)."""
    S = _make_settings()
    W, H = 64, 64
    active = {0, 1, 2, 3, 4, 5}

    inside = _build_inside_mask(W, H, S)
    buffer = compute_edge_buffer_mask(W, H, S, 0.2, 2, active)

    combined = buffer.any(dim=0)
    assert not (combined & ~inside).any(), "Buffer pixels found in waste area"
    print("PASS: test_buffer_inside_hex_only")


if __name__ == "__main__":
    test_buffer_depth_0_is_subset_of_border()
    test_buffer_depth_2_superset_of_depth_0()
    test_buffer_only_active_directions()
    test_buffer_inside_hex_only()
    print("\nAll tests passed!")
