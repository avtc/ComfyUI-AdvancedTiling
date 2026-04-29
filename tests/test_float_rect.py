import sys
import os
import math

_test_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.dirname(_test_dir)
_custom_nodes = os.path.dirname(_pkg_dir)
sys.path.insert(0, _custom_nodes)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ComfyUI_AdvancedTiling",
    os.path.join(_pkg_dir, "__init__.py"),
    submodule_search_locations=[_pkg_dir],
)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["ComfyUI_AdvancedTiling"] = _pkg
sys.modules["ComfyUI-AdvancedTiling"] = _pkg
_spec.loader.exec_module(_pkg)

from ComfyUI_AdvancedTiling.modes import Settings
from ComfyUI_AdvancedTiling.modes.rect import compute_float_rect_dims, rect_tiling


def test_no_margin_no_divisible():
    """scale=1.0, min_margin=0, divisible_by=1 → dims equal input."""
    s = Settings("Rectangular", 0.0, scale=1.0, min_margin=0, divisible_by=1)
    w, h = compute_float_rect_dims(128, 128, s)
    assert w == 128.0
    assert h == 128.0


def test_scale_reduces_dims():
    """scale=0.5 → dims are half."""
    s = Settings("Rectangular", 0.0, scale=0.5, min_margin=0, divisible_by=1)
    w, h = compute_float_rect_dims(128, 128, s)
    assert w == 64.0
    assert h == 64.0


def test_min_margin_reduces_dims():
    """scale=1.0, min_margin=4 → dims reduced by 2*margin."""
    s = Settings("Rectangular", 0.0, scale=1.0, min_margin=4, divisible_by=1)
    w, h = compute_float_rect_dims(128, 128, s)
    assert w == 120.0
    assert h == 120.0


def test_scale_and_margin():
    """Both scale and margin applied."""
    s = Settings("Rectangular", 0.0, scale=0.8, min_margin=4, divisible_by=1)
    w, h = compute_float_rect_dims(128, 128, s)
    assert w == 128.0 * 0.8 - 2 * 4  # 94.4
    assert h == 128.0 * 0.8 - 2 * 4


def test_divisible_by_rounds_down():
    """divisible_by rounds dims down to nearest valid image-pixel multiple."""
    s = Settings("Rectangular", 0.0, scale=0.8, min_margin=4, divisible_by=64)
    w, h = compute_float_rect_dims(128, 128, s, vae_factor=8)
    assert w == 88.0
    assert h == 88.0


def test_divisible_by_no_rounding_when_exact():
    """divisible_by=8 with dims already divisible → no change."""
    s = Settings("Rectangular", 0.0, scale=0.5, min_margin=0, divisible_by=8)
    w, h = compute_float_rect_dims(128, 128, s, vae_factor=8)
    assert w == 64.0
    assert h == 64.0


def test_non_square():
    """Non-square dimensions."""
    s = Settings("Rectangular", 0.0, scale=0.75, min_margin=2, divisible_by=1)
    w, h = compute_float_rect_dims(100, 80, s)
    assert w == 100.0 * 0.75 - 2 * 2  # 71.0
    assert h == 80.0 * 0.75 - 2 * 2   # 56.0


# --- rect_tiling tests ---

def test_rect_identity_inside():
    """Pixels inside the float rectangle map to themselves."""
    s = Settings("Rectangular", 0.0, scale=1.0, min_margin=0, divisible_by=1)
    result = rect_tiling(5, 5, (10, 10), (10, 10), s)
    assert result == (5, 5)


def test_rect_wraps_right_to_left():
    """Pixel at right edge wraps to left."""
    s = Settings("Rectangular", 0.0, scale=1.0, min_margin=0, divisible_by=1)
    result = rect_tiling(10, 5, (10, 10), (10, 10), s)
    assert result == (0, 5)


def test_rect_wraps_bottom_to_top():
    """Pixel at bottom edge wraps to top."""
    s = Settings("Rectangular", 0.0, scale=1.0, min_margin=0, divisible_by=1)
    result = rect_tiling(5, 10, (10, 10), (10, 10), s)
    assert result == (5, 0)


def test_rect_scaled_wraps_outside():
    """Pixel outside scaled rectangle wraps to inside."""
    s = Settings("Rectangular", 0.0, scale=0.5, min_margin=0, divisible_by=1)
    # work_w = 10 * 0.5 = 5.0, center = 5.0
    # x=0 is outside the rect (rect starts at 2.5), so it wraps
    result = rect_tiling(0, 0, (10, 10), (10, 10), s)
    assert result != (0, 0)  # Should wrap to inside


def test_rect_scaled_identity_inside():
    """Pixel inside scaled rectangle maps to itself."""
    s = Settings("Rectangular", 0.0, scale=0.5, min_margin=0, divisible_by=1)
    # work_w = 5.0, center = 5.0, rect spans [2.5, 7.5)
    # x=5 is inside
    result = rect_tiling(5, 5, (10, 10), (10, 10), s)
    assert result == (5, 5)


# --- calculate_mapping tests ---

import ComfyUI_AdvancedTiling.advanced_tiling as at_mod


def test_calculate_mapping_rect_identity():
    """All pixels inside rect map to themselves → empty mapping."""
    s = Settings("Rectangular", 0.0, scale=1.0, min_margin=0, divisible_by=1)
    src_x, src_y, new_x, new_y = at_mod.calculate_mapping((10, 10), (10, 10), s)
    assert len(src_x) == 0


def test_calculate_mapping_rect_scaled_has_remapping():
    """Scaled rect produces non-empty mapping for outside pixels."""
    s = Settings("Rectangular", 0.0, scale=0.5, min_margin=0, divisible_by=1)
    src_x, src_y, new_x, new_y = at_mod.calculate_mapping((10, 10), (10, 10), s)
    assert len(src_x) > 0


# --- create_crop_mask tests ---


def test_crop_mask_rect_full():
    """scale=1.0, no margin → all pixels in mask."""
    s = Settings("Rectangular", 0.0, scale=1.0, min_margin=0, divisible_by=1)
    mask = at_mod.create_crop_mask(10, 10, s)
    assert mask.shape == (1, 10, 10, 1)
    assert mask.sum().item() == 100  # all ones


def test_crop_mask_rect_scaled():
    """scale=0.5 → mask has fewer pixels than total."""
    s = Settings("Rectangular", 0.0, scale=0.5, min_margin=0, divisible_by=1)
    mask = at_mod.create_crop_mask(10, 10, s)
    assert mask.shape == (1, 10, 10, 1)
    assert mask.sum().item() < 100  # some pixels outside rect
    assert mask.sum().item() > 0    # some pixels inside


# --- integration tests ---


def test_end_to_end_scale_and_divisible():
    """Full pipeline: scale → min_margin → divisible_by → wrapping → crop mask."""
    W, H = 128, 128
    s = Settings("Rectangular", 0.0, scale=0.8, min_margin=4, divisible_by=64)

    # 1. Float dims
    work_w, work_h = compute_float_rect_dims(W, H, s, vae_factor=8)
    assert work_w > 0 and work_h > 0
    assert (work_w * 8) % 64 == 0
    assert (work_h * 8) % 64 == 0

    # 2. Wrapping: center pixel is identity
    result = rect_tiling(W // 2, H // 2, (W, H), (W, H), s)
    assert result == (W // 2, H // 2)

    # 3. Wrapping: corner pixel wraps
    result = rect_tiling(0, 0, (W, H), (W, H), s)
    assert result != (0, 0)

    # 4. Mapping is non-empty
    mapping = at_mod.calculate_mapping((W, H), (W, H), s)
    assert len(mapping[0]) > 0

    # 5. Crop mask covers fewer pixels than total
    mask = at_mod.create_crop_mask(W, H, s)
    assert mask.sum().item() < W * H
    assert mask.sum().item() > 0


# --- edge case tests ---


def test_zero_scale_clamps_to_minimum():
    """scale=0.0 → dims clamped to 1.0 (no ZeroDivisionError)."""
    s = Settings("Rectangular", 0.0, scale=0.0, min_margin=0, divisible_by=1)
    w, h = compute_float_rect_dims(128, 128, s)
    assert w >= 1.0
    assert h >= 1.0


def test_negative_dims_clamped():
    """Large min_margin relative to dims → clamped to 1.0."""
    s = Settings("Rectangular", 0.0, scale=0.1, min_margin=20, divisible_by=1)
    w, h = compute_float_rect_dims(32, 32, s)
    assert w >= 1.0
    assert h >= 1.0


def test_zero_scale_no_crash_in_tiling():
    """scale=0.0 does not cause ZeroDivisionError in rect_tiling."""
    s = Settings("Rectangular", 0.0, scale=0.0, min_margin=0, divisible_by=1)
    result = rect_tiling(5, 5, (10, 10), (10, 10), s)
    assert isinstance(result, tuple) and len(result) == 2


def test_non_square_calculate_mapping():
    """calculate_mapping works for non-square dimensions."""
    s = Settings("Rectangular", 0.0, scale=0.5, min_margin=0, divisible_by=1)
    src_x, src_y, new_x, new_y = at_mod.calculate_mapping((20, 10), (20, 10), s)
    assert len(src_x) > 0
    assert len(src_x) == len(src_y) == len(new_x) == len(new_y)


def test_non_square_create_crop_mask():
    """create_crop_mask works for non-square dimensions."""
    s = Settings("Rectangular", 0.0, scale=0.5, min_margin=0, divisible_by=1)
    mask = at_mod.create_crop_mask(20, 10, s)
    assert mask.shape == (1, 10, 20, 1)
    assert mask.sum().item() < 20 * 10
    assert mask.sum().item() > 0


def test_tiling_vs_mapping_consistency():
    """Per-pixel rect_tiling and vectorized calculate_mapping produce same results."""
    W, H = 16, 16
    s = Settings("Rectangular", 0.0, scale=0.6, min_margin=2, divisible_by=1)
    src_x, src_y, new_x, new_y = at_mod.calculate_mapping((W, H), (W, H), s)

    for i in range(len(src_x)):
        sx, sy = src_x[i].item(), src_y[i].item()
        nx, ny = new_x[i].item(), new_y[i].item()
        px, py = rect_tiling(sx, sy, (W, H), (W, H), s)
        assert (px, py) == (nx, ny), f"Mismatch at ({sx},{sy}): rect_tiling={px},{py} vs mapping={nx},{ny}"


def test_divisible_by_not_multiple_of_vae_factor():
    """divisible_by that isn't a multiple of vae_factor still produces valid dims."""
    s = Settings("Rectangular", 0.0, scale=1.0, min_margin=0, divisible_by=12)
    w, h = compute_float_rect_dims(64, 64, s, vae_factor=8)
    assert w > 0 and h > 0
    # 12/8 = 1.5, so dims should be multiples of 1.5
    assert w % 1.5 == 0.0


if __name__ == "__main__":
    import sys
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
