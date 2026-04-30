"""
Test geometry consistency across all tiling subsystems.

Compares four subsystems for each mode and settings combo:
1. Latent wrapping (calculate_mapping) -- vectorized Conv2d path
2. Per-pixel tiling (rect_tiling / hex_tiling) -- used by toroidal attention
3. Toroidal attention boundary (_compute_*_boundary_pairs) -- DiT K/V injection
4. VAE decode crop (create_crop_mask) -- post-decode cropping

Tests the full parameter matrix:
  scale = {1.0, 0.9}
  min_margin = {-1, 0, 4}  (-1 is auto sentinel, resolved before use)
  divisible_by = {1, 16}
  mode = {Rectangular, Hexagon}

Prints ASCII visualizations and asserts all four agree on the working area.
"""

import sys
import os

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

import torch
import numpy as np
from ComfyUI_AdvancedTiling.modes import Settings
from ComfyUI_AdvancedTiling.modes.rect import compute_float_rect_dims, rect_tiling
from ComfyUI_AdvancedTiling.modes.hex import compute_float_hex_size, hex_tiling
import ComfyUI_AdvancedTiling.advanced_tiling as at_mod
import ComfyUI_AdvancedTiling.toroidal_attention as ta_mod

VAE_FACTOR = 8


# ---------------------------------------------------------------------------
# Helper functions -- call actual code, no logic duplication
# ---------------------------------------------------------------------------

def build_wrapping_identity(W, H, settings, vae_factor):
    """2D bool array: True where pixel maps to itself (inside working area)."""
    src_x, src_y, new_x, new_y = at_mod.calculate_mapping(
        (W, H), (W, H), settings, vae_factor,
    )
    identity = torch.ones((H, W), dtype=torch.bool)
    if len(src_x) > 0:
        identity[src_y, src_x] = False
    return identity


def build_tiling_identity(W, H, settings, vae_factor, mode):
    """2D bool array from per-pixel tiling: True where pixel maps to itself."""
    identity = np.ones((H, W), dtype=bool)
    tiling_fn = rect_tiling if mode == "Rectangular" else hex_tiling
    for y in range(H):
        for x in range(W):
            nx, ny = tiling_fn(x, y, (W, H), (W, H), settings, vae_factor)
            if nx != x or ny != y:
                identity[y, x] = False
    return identity


def build_crop_inside(W, H, settings, vae_factor):
    """2D bool array from crop mask: True where mask is 1."""
    mask = at_mod.create_crop_mask(W, H, settings, vae_factor)
    return mask[0, :, :, 0].bool()


def build_boundary_map(H, W, settings, vae_factor, mode):
    """2D bool array: True for patches on the toroidal attention boundary."""
    if mode == "Rectangular":
        fn = ta_mod._compute_rect_boundary_pairs
    else:
        fn = ta_mod._compute_hex_boundary_pairs

    boundary_idx, source_idx, off_h, off_w = fn(H, W, settings, vae_factor)
    boundary = np.zeros((H, W), dtype=bool)
    if len(boundary_idx) > 0:
        for idx in boundary_idx.tolist():
            by, bx = divmod(idx, W)
            if 0 <= by < H and 0 <= bx < W:
                boundary[by, bx] = True
    return boundary


def ascii_grid(bool_2d, inside='#', outside='.'):
    """Convert 2D bool array to ASCII string."""
    lines = []
    for row in bool_2d:
        lines.append(''.join(inside if v else outside for v in row))
    return '\n'.join(lines)


def ascii_composite(identity, boundary, inside='#', boundary_ch='B', outside='.'):
    """Show identity map with boundary patches highlighted."""
    lines = []
    for y in range(identity.shape[0]):
        row = []
        for x in range(identity.shape[1]):
            if boundary[y, x]:
                row.append(boundary_ch)
            elif bool(identity[y, x]):
                row.append(inside)
            else:
                row.append(outside)
        lines.append(''.join(row))
    return '\n'.join(lines)


def ascii_overlay(a, b):
    """Show mismatch between two bool maps. 1=a-only, 2=b-only, ==agree."""
    lines = []
    for y in range(a.shape[0]):
        row = []
        for x in range(a.shape[1]):
            va, vb = bool(a[y, x]), bool(b[y, x])
            if va == vb:
                row.append('=')
            elif va:
                row.append('1')
            else:
                row.append('2')
        lines.append(''.join(row))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Core test function
# ---------------------------------------------------------------------------

def check_geometry(label, mode, scale, min_margin, div_by, W, H, vae_factor,
                   print_ascii=True, assert_consistency=True):
    """Run full geometry consistency check for one mode/settings combo."""
    settings = Settings(mode, 0.0, scale=scale, min_margin=min_margin, divisible_by=div_by)

    if mode == "Rectangular":
        work_w, work_h = compute_float_rect_dims(W, H, settings, vae_factor)
        dim_info = f"work_w={work_w:.2f}, work_h={work_h:.2f}"
    else:
        hex_size = compute_float_hex_size(W, H, settings, vae_factor)
        dim_info = f"hex_size={hex_size:.2f}, hex_height={hex_size*2:.2f}"

    wrap_id = build_wrapping_identity(W, H, settings, vae_factor)
    tile_id = build_tiling_identity(W, H, settings, vae_factor, mode)
    crop_id = build_crop_inside(W, H, settings, vae_factor)
    boundary = build_boundary_map(H, W, settings, vae_factor, mode)

    wrap_np = wrap_id.numpy() if isinstance(wrap_id, torch.Tensor) else wrap_id
    tile_np = tile_id.numpy() if isinstance(tile_id, torch.Tensor) else tile_id
    crop_np = crop_id.numpy() if isinstance(crop_id, torch.Tensor) else crop_id

    wrap_count = int(wrap_np.sum())
    tile_count = int(tile_np.sum())
    crop_count = int(crop_np.sum())
    boundary_count = int(boundary.sum())

    if print_ascii:
        print(f"\n{'='*60}")
        print(f"{label}: {mode} scale={scale} min_margin={min_margin} div_by={div_by}")
        print(f"Grid: {W}x{H}, vae_factor={vae_factor}, {dim_info}")
        print(f"Inside counts: wrap={wrap_count} tile={tile_count} crop={crop_count} boundary={boundary_count}")
        print(f"{'='*60}")

        print(f"\n1. Wrapping identity:  #={wrap_count} .={W*H - wrap_count}")
        print(ascii_grid(wrap_np))

        print(f"\n2. Per-pixel tiling:  #={tile_count}")
        print(ascii_grid(tile_np))

        print(f"\n3. Crop mask:  #={crop_count}")
        print(ascii_grid(crop_np))

        print(f"\n4. Toroidal boundary:  B={boundary_count}")
        print(ascii_composite(wrap_np, boundary))

        if not np.array_equal(wrap_np, crop_np):
            print(f"\nDIFF wrapping vs crop:  1=wrap-only  2=crop-only")
            print(ascii_overlay(wrap_np, crop_np))
        else:
            print(f"\nwrapping == crop: MATCH")

        if not np.array_equal(wrap_np, tile_np):
            print(f"\nDIFF wrapping vs tiling:  1=wrap-only  2=tile-only")
            print(ascii_overlay(wrap_np, tile_np))
        else:
            print(f"\nwrapping == tiling: MATCH")

    errors = []

    if not np.array_equal(wrap_np, tile_np):
        diff = int((wrap_np != tile_np).sum())
        errors.append(f"wrapping != tiling: {diff} pixels differ")

    if not np.array_equal(wrap_np, crop_np):
        diff = int((wrap_np != crop_np).sum())
        errors.append(f"wrapping != crop: {diff} pixels differ")

    boundary_outside = boundary & ~wrap_np
    if boundary_outside.any():
        count = int(boundary_outside.sum())
        errors.append(f"boundary not subset of wrapping inside: {count} boundary patches outside")

    has_outside = int((~wrap_np).sum()) > 0
    if has_outside and boundary_count == 0:
        errors.append("has outside pixels but no toroidal boundary patches")

    if mode == "Rectangular":
        fn = ta_mod._compute_rect_boundary_pairs
    else:
        fn = ta_mod._compute_hex_boundary_pairs
    boundary_idx, source_idx, _, _ = fn(H, W, settings, vae_factor)
    if has_outside and len(source_idx) == 0:
        errors.append("has outside pixels but no toroidal source pairs")

    if assert_consistency and errors:
        raise AssertionError('\n'.join(errors))

    return errors


# ---------------------------------------------------------------------------
# Parameter matrix: scale x min_margin x div_by x mode (at realistic 128x128 latent)
# ---------------------------------------------------------------------------

PARAM_GRID = [
    (1.0, 0, 1),
    (1.0, 0, 16),
    (1.0, 4, 1),
    (1.0, 4, 16),
    (0.9, 0, 1),
    (0.9, 0, 16),
    (0.9, 4, 1),
    (0.9, 4, 16),
]


# Generate test functions for the full matrix at realistic 128x128 latent
for _mode in ["Rectangular", "Hexagon"]:
    for _scale, _margin, _div in PARAM_GRID:
        _tag = f"{_mode[:4].lower()}_s{_scale}_m{_margin}_d{_div}"
        def _make_test(mode, scale, margin, div_by):
            def test_fn():
                label = f"{mode[:4]}-s{scale}-m{margin}-d{div_by}"
                check_realistic_borders(label, mode, scale, margin, div_by)
                # Also verify all 4 subsystems agree at latent resolution
                errors = check_geometry(
                    label, mode, scale, margin, div_by,
                    W=128, H=128, vae_factor=1,
                    print_ascii=False, assert_consistency=True,
                )
                assert not errors, '\n'.join(errors)
            return test_fn
        _fn = _make_test(_mode, _scale, _margin, _div)
        _fn.__name__ = f"test_{_tag}"
        globals()[_fn.__name__] = _fn


# ---------------------------------------------------------------------------
# Auto-resolution tests (min_margin=-1)
# ---------------------------------------------------------------------------

def test_auto_resolve_dit_rect():
    """min_margin=-1, scale=0 resolves to scale=7/8, min_margin=4 for DiT Rect."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=1)
    resolved = raw._resolve_auto(is_conv2d=False)

    assert resolved.scale == 7/8
    assert resolved.min_margin == 4
    assert raw.scale == 0.0, "original should not be mutated"
    assert raw.min_margin == -1

    # Full realistic test with visualization
    check_realistic_borders("RECT-auto-DiT", "Rectangular",
                            resolved.scale, resolved.min_margin, resolved.divisible_by)
    errors = check_geometry(
        "RECT-auto-DiT", "Rectangular",
        scale=resolved.scale, min_margin=resolved.min_margin, div_by=resolved.divisible_by,
        W=128, H=128, vae_factor=1,
        print_ascii=False, assert_consistency=True,
    )
    assert not errors, '\n'.join(errors)


def test_auto_resolve_conv2d_rect():
    """min_margin=-1, scale=0 resolves to scale=1.0, min_margin=0 for Conv2d Rect."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=1)
    resolved = raw._resolve_auto(is_conv2d=True)

    assert resolved.scale == 1.0
    assert resolved.min_margin == 0

    check_realistic_borders("RECT-auto-Conv2d", "Rectangular",
                            resolved.scale, resolved.min_margin, resolved.divisible_by)
    errors = check_geometry("RECT-auto-Conv2d", "Rectangular",
                            resolved.scale, resolved.min_margin, resolved.divisible_by,
                            W=128, H=128, vae_factor=1, print_ascii=False)
    assert not errors, '\n'.join(errors)


def test_auto_resolve_dit_hex():
    """min_margin=-1, scale=0 resolves to scale=1.0, min_margin=0 for DiT Hex."""
    raw = Settings("Hexagon", 0.0, scale=0.0, min_margin=-1, divisible_by=1)
    resolved = raw._resolve_auto(is_conv2d=False)

    assert resolved.scale == 1.0
    assert resolved.min_margin == 0

    check_realistic_borders("HEX-auto-DiT", "Hexagon",
                            resolved.scale, resolved.min_margin, resolved.divisible_by)
    errors = check_geometry("HEX-auto-DiT", "Hexagon",
                            resolved.scale, resolved.min_margin, resolved.divisible_by,
                            W=128, H=128, vae_factor=1, print_ascii=False)
    assert not errors, '\n'.join(errors)


def test_auto_resolve_conv2d_hex():
    """min_margin=-1, scale=0 resolves to scale=1.0, min_margin=0 for Conv2d Hex."""
    raw = Settings("Hexagon", 0.0, scale=0.0, min_margin=-1, divisible_by=1)
    resolved = raw._resolve_auto(is_conv2d=True)

    assert resolved.scale == 1.0
    assert resolved.min_margin == 0

    check_realistic_borders("HEX-auto-Conv2d", "Hexagon",
                            resolved.scale, resolved.min_margin, resolved.divisible_by)
    errors = check_geometry("HEX-auto-Conv2d", "Hexagon",
                            resolved.scale, resolved.min_margin, resolved.divisible_by,
                            W=128, H=128, vae_factor=1, print_ascii=False)
    assert not errors, '\n'.join(errors)


def test_auto_resolve_dit_rect_div16():
    """min_margin=-1, scale=0, div_by=16 resolves for DiT Rect."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=16)
    resolved = raw._resolve_auto(is_conv2d=False)

    assert resolved.scale == 7/8
    assert resolved.min_margin == 4

    check_realistic_borders("RECT-auto-DiT-d16", "Rectangular",
                            resolved.scale, resolved.min_margin, resolved.divisible_by)
    errors = check_geometry("RECT-auto-DiT-d16", "Rectangular",
                            resolved.scale, resolved.min_margin, resolved.divisible_by,
                            W=128, H=128, vae_factor=1, print_ascii=False)
    assert not errors, '\n'.join(errors)


def test_auto_resolve_dit_hex_div16():
    """min_margin=-1, scale=0, div_by=16 resolves for DiT Hex."""
    raw = Settings("Hexagon", 0.0, scale=0.0, min_margin=-1, divisible_by=16)
    resolved = raw._resolve_auto(is_conv2d=False)

    assert resolved.scale == 1.0
    assert resolved.min_margin == 0

    check_realistic_borders("HEX-auto-DiT-d16", "Hexagon",
                            resolved.scale, resolved.min_margin, resolved.divisible_by)
    errors = check_geometry("HEX-auto-DiT-d16", "Hexagon",
                            resolved.scale, resolved.min_margin, resolved.divisible_by,
                            W=128, H=128, vae_factor=1, print_ascii=False)
    assert not errors, '\n'.join(errors)


# ---------------------------------------------------------------------------
# Auto-resolution with explicit scale=0.9 + min_margin=-1
# ---------------------------------------------------------------------------

def test_auto_resolve_dit_rect_s09_mneg1():
    """scale=0.9, min_margin=-1, div_by=16 for DiT Rect: margin resolves -1->4."""
    raw = Settings("Rectangular", 0.0, scale=0.9, min_margin=-1, divisible_by=16)
    resolved = raw._resolve_auto(is_conv2d=False)

    assert resolved.scale == 0.9
    assert resolved.min_margin == 4

    check_realistic_borders("RECT-s0.9-auto-DiT", "Rectangular",
                            resolved.scale, resolved.min_margin, resolved.divisible_by)
    errors = check_geometry("RECT-s0.9-auto-DiT", "Rectangular",
                            resolved.scale, resolved.min_margin, resolved.divisible_by,
                            W=128, H=128, vae_factor=1, print_ascii=False)
    assert not errors, '\n'.join(errors)


def test_auto_resolve_conv2d_rect_s09_mneg1():
    """scale=0.9, min_margin=-1, div_by=16 for Conv2d Rect: margin resolves -1->0."""
    raw = Settings("Rectangular", 0.0, scale=0.9, min_margin=-1, divisible_by=16)
    resolved = raw._resolve_auto(is_conv2d=True)

    assert resolved.scale == 0.9
    assert resolved.min_margin == 0

    check_realistic_borders("RECT-s0.9-auto-Conv2d", "Rectangular",
                            resolved.scale, resolved.min_margin, resolved.divisible_by)
    errors = check_geometry("RECT-s0.9-auto-Conv2d", "Rectangular",
                            resolved.scale, resolved.min_margin, resolved.divisible_by,
                            W=128, H=128, vae_factor=1, print_ascii=False)
    assert not errors, '\n'.join(errors)


# ---------------------------------------------------------------------------
# Auto-resolution HEX with scale=0.9, min_margin=-1, div_by=16
# ---------------------------------------------------------------------------

def test_auto_resolve_hex_s09_mneg1_dit():
    """HEX scale=0.9, min_margin=-1, div_by=16 for DiT: margin resolves -1->0."""
    raw = Settings("Hexagon", 0.0, scale=0.9, min_margin=-1, divisible_by=16)
    resolved = raw._resolve_auto(is_conv2d=False)

    assert resolved.scale == 0.9
    assert resolved.min_margin == 0

    check_realistic_borders("HEX-s0.9-auto-DiT", "Hexagon",
                            resolved.scale, resolved.min_margin, resolved.divisible_by)
    errors = check_geometry("HEX-s0.9-auto-DiT", "Hexagon",
                            resolved.scale, resolved.min_margin, resolved.divisible_by,
                            W=128, H=128, vae_factor=1, print_ascii=False)
    assert not errors, '\n'.join(errors)


def test_auto_resolve_hex_s09_mneg1_conv2d():
    """HEX scale=0.9, min_margin=-1, div_by=16 for Conv2d: margin resolves -1->0."""
    raw = Settings("Hexagon", 0.0, scale=0.9, min_margin=-1, divisible_by=16)
    resolved = raw._resolve_auto(is_conv2d=True)

    assert resolved.scale == 0.9
    assert resolved.min_margin == 0

    check_realistic_borders("HEX-s0.9-auto-Conv2d", "Hexagon",
                            resolved.scale, resolved.min_margin, resolved.divisible_by)
    errors = check_geometry("HEX-s0.9-auto-Conv2d", "Hexagon",
                            resolved.scale, resolved.min_margin, resolved.divisible_by,
                            W=128, H=128, vae_factor=1, print_ascii=False)
    assert not errors, '\n'.join(errors)


# ---------------------------------------------------------------------------
# Realistic resolution: border visualization
# ---------------------------------------------------------------------------

def ascii_border_region(bool_2d, border_px=5):
    """Show only the border region of a large grid as ASCII.

    Prints top, bottom, left, right edges (border_px wide) plus a center crop.
    """
    H, W = bool_2d.shape
    lines = []

    # Top border
    for y in range(min(border_px, H)):
        lines.append(f"  y={y:3d}  " + ''.join('#' if v else '.' for v in bool_2d[y]))

    if H > 2 * border_px:
        lines.append(f"  ... ({H - 2*border_px} rows omitted)")
        # Bottom border
        for y in range(max(border_px, H - border_px), H):
            lines.append(f"  y={y:3d}  " + ''.join('#' if v else '.' for v in bool_2d[y]))

    return '\n'.join(lines)


def ascii_boundary_zoom(bool_2d, zoom_px=10):
    """Zoom into the boundary transition zone of a large bool array.

    Finds the first row/col where inside transitions to outside (or vice versa)
    and shows a zoom_px x zoom_px window around it. Shows both top-left and
    bottom-right corners of the working area.
    """
    H, W = bool_2d.shape
    lines = []

    # Find top-left boundary: first row that has a mix of inside/outside
    for y in range(H):
        row = bool_2d[y]
        if row.any() and not row.all():
            # Found a boundary row -- zoom in around the transition
            x_trans = 0
            for x in range(W):
                if bool(row[x]) != bool(row[0]):
                    x_trans = x
                    break
            y0 = max(0, y - 2)
            x0 = max(0, x_trans - zoom_px // 2)
            lines.append(f"  Top-left boundary (row {y0}):")
            for ry in range(y0, min(H, y0 + zoom_px)):
                lines.append(f"    y={ry:4d}  " + ''.join(
                    '#' if bool_2d[ry, cx] else '.'
                    for cx in range(x0, min(W, x0 + zoom_px * 2))
                ))
            break

    # Find bottom-right boundary
    for y in range(H - 1, -1, -1):
        row = bool_2d[y]
        if row.any() and not row.all():
            x_trans = W - 1
            for x in range(W - 1, -1, -1):
                if bool(row[x]) != bool(row[-1]):
                    x_trans = x
                    break
            y1 = min(H, y + 3)
            x1 = max(0, x_trans - zoom_px)
            lines.append(f"  Bottom-right boundary (row {y1 - zoom_px}):")
            for ry in range(max(0, y1 - zoom_px), y1):
                lines.append(f"    y={ry:4d}  " + ''.join(
                    '#' if bool_2d[ry, cx] else '.'
                    for cx in range(x1, min(W, x1 + zoom_px * 2))
                ))
            break

    return '\n'.join(lines)


def check_realistic_borders(label, mode, scale, min_margin, div_by,
                             W_lat=128, H_lat=128, vae_factor=VAE_FACTOR):
    """Check geometry at realistic latent resolution with border visualization."""
    settings = Settings(mode, 0.0, scale=scale, min_margin=min_margin, divisible_by=div_by)

    if mode == "Rectangular":
        work_w, work_h = compute_float_rect_dims(W_lat, H_lat, settings, vae_factor)
        dim_info = f"work_w={work_w:.2f}, work_h={work_h:.2f}"
    else:
        hex_size = compute_float_hex_size(W_lat, H_lat, settings, vae_factor)
        dim_info = f"hex_size={hex_size:.2f}, hex_height={hex_size*2:.2f}"

    print(f"\n{'='*70}")
    print(f"{label}: {mode} scale={scale} min_margin={min_margin} div_by={div_by}")
    print(f"Latent: {W_lat}x{H_lat}, vae_factor={vae_factor}, {dim_info}")
    print(f"{'='*70}")

    # Wrapping identity at latent resolution
    wrap_id = build_wrapping_identity(W_lat, H_lat, settings, vae_factor)
    wrap_np = wrap_id.numpy() if isinstance(wrap_id, torch.Tensor) else wrap_id

    inside_count = int(wrap_np.sum())
    outside_count = W_lat * H_lat - inside_count
    print(f"  Inside: {inside_count}/{W_lat*H_lat} ({inside_count/(W_lat*H_lat)*100:.1f}%)")

    # Border visualization at latent resolution
    print(f"\n  Latent wrapping border (top/bottom 5 rows of {W_lat}x{H_lat}):")
    print(ascii_border_region(wrap_np, border_px=5))

    # Crop mask at image resolution
    img_w, img_h = W_lat * vae_factor, H_lat * vae_factor
    mask = at_mod.create_crop_mask(img_w, img_h, settings, vae_factor)
    rmin, rmax, cmin, cmax = at_mod._mask_bounding_box(mask)

    if mode == "Hexagon":
        sq_rmin, sq_rmax, sq_cmin, sq_cmax = at_mod.hex_square_crop(
            rmin, rmax, cmin, cmax, img_h, img_w, div_by,
        )
        crop_w = sq_cmax - sq_cmin + 1
        crop_h = sq_rmax - sq_rmin + 1
    else:
        crop_w = cmax - cmin + 1
        crop_h = rmax - rmin + 1

    print(f"\n  Image crop: {crop_w}x{crop_h} (from {img_w}x{img_h})")

    # Image-resolution boundary visualization (crop mask)
    crop_mask_np = mask[0, :, :, 0].bool().numpy()
    if outside_count > 0:
        print(f"\n  Image crop mask boundary zoom ({img_w}x{img_h}, vae_factor={vae_factor}):")
        print(ascii_boundary_zoom(crop_mask_np, zoom_px=8))

    if mode == "Hexagon":
        # Hex crop is always square: width == height (hex bounding box height)
        assert crop_w == crop_h, f"hex crop not square: {crop_w}x{crop_h}"
        print(f"  Hex square crop: {crop_w}x{crop_h}")
        if div_by > 1:
            assert crop_w % div_by == 0, f"hex crop {crop_w} not div by {div_by}"
            print(f"  div_by={div_by} check: PASS ({crop_w}%{div_by}==0)")
    elif div_by > 1:
        assert crop_w % div_by == 0, f"crop_w={crop_w} not div by {div_by}"
        assert crop_h % div_by == 0, f"crop_h={crop_h} not div by {div_by}"
        print(f"  div_by={div_by} check: PASS ({crop_w}%{div_by}==0, {crop_h}%{div_by}==0)")

    # Verify wrapping and crop agree on inside count
    crop_id = mask[0, :, :, 0].bool()
    crop_inside = int(crop_id.sum())
    expected_crop_inside = inside_count * (vae_factor ** 2)
    # Allow 2*vae_factor tolerance per edge for float rounding
    tolerance = 2 * vae_factor * (W_lat + H_lat)
    assert abs(crop_inside - expected_crop_inside) <= tolerance, (
        f"crop inside ({crop_inside}) too far from expected ({expected_crop_inside}"
        f" +- {tolerance})"
    )
    print(f"  Crop inside: {crop_inside} (expected ~{expected_crop_inside}, tolerance {tolerance})")

    return True


# ---------------------------------------------------------------------------
# Realistic resolution with auto-resolved settings
# ---------------------------------------------------------------------------

def test_rect_realistic_dit_auto():
    """DiT Rect auto-resolved at 1024x1024."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=1)
    settings = raw._resolve_auto(is_conv2d=False)
    check_realistic_borders("RECT-DiT-auto", "Rectangular",
                            settings.scale, settings.min_margin, settings.divisible_by)


def test_rect_realistic_dit_auto_d16():
    """DiT Rect auto-resolved + div_by=16 at 1024x1024."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=16)
    settings = raw._resolve_auto(is_conv2d=False)
    check_realistic_borders("RECT-DiT-auto-d16", "Rectangular",
                            settings.scale, settings.min_margin, settings.divisible_by)


def test_hex_realistic_dit_auto():
    """DiT Hex auto-resolved at 1024x1024."""
    raw = Settings("Hexagon", 0.0, scale=0.0, min_margin=-1, divisible_by=1)
    settings = raw._resolve_auto(is_conv2d=False)
    check_realistic_borders("HEX-DiT-auto", "Hexagon",
                            settings.scale, settings.min_margin, settings.divisible_by)


def test_conv2d_realistic_auto():
    """Conv2d auto-resolved at 1024x1024 (VAE decode path -- was buggy)."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=1)
    settings = raw._resolve_auto(is_conv2d=True)
    assert settings.scale == 1.0
    assert settings.min_margin == 0

    work_w, work_h = compute_float_rect_dims(128, 128, settings, VAE_FACTOR)
    assert work_w == 128.0, f"expected 128.0, got {work_w}"

    mask = at_mod.create_crop_mask(1024, 1024, settings, vae_factor=VAE_FACTOR)
    assert mask.sum().item() == 1024 * 1024, "Conv2d auto-resolved should cover full image"
    print(f"\nConv2d RECT auto: scale={settings.scale}, min_margin={settings.min_margin}, full image")


# ---------------------------------------------------------------------------
# Margin in latent pixels verification
# ---------------------------------------------------------------------------

def test_margin_in_latent_pixels_rect():
    """min_margin is in latent pixels: float dims identical regardless of vae_factor."""
    s = Settings("Rectangular", 0.0, scale=0.9, min_margin=4, divisible_by=1)

    work_w_direct, _ = compute_float_rect_dims(128, 128, s, vae_factor=1)
    work_w_img, _ = compute_float_rect_dims(128, 128, s, vae_factor=8)

    print(f"RECT margin in latent pixels: direct={work_w_direct} img_path={work_w_img}")
    assert work_w_direct == work_w_img

    mask = at_mod.create_crop_mask(1024, 1024, s, vae_factor=8)
    rmin, rmax, cmin, cmax = at_mod._mask_bounding_box(mask)
    crop_w = cmax - cmin + 1
    expected_w = int(work_w_direct * 8)
    print(f"  crop_w={crop_w}, expected={expected_w}")
    assert abs(crop_w - expected_w) <= 1


def test_margin_in_latent_pixels_hex():
    """min_margin is in latent pixels for hex mode too."""
    s = Settings("Hexagon", 0.0, scale=0.9, min_margin=2, divisible_by=1)

    size_direct = compute_float_hex_size(128, 128, s, vae_factor=1)
    size_img = compute_float_hex_size(128, 128, s, vae_factor=8)

    print(f"HEX margin in latent pixels: direct={size_direct} img_path={size_img}")
    assert size_direct == size_img


# ---------------------------------------------------------------------------
# Proportionality checks
# ---------------------------------------------------------------------------

def test_float_dim_proportionality():
    """Float dims scale proportionally when min_margin=0."""
    for mode in ["Rectangular", "Hexagon"]:
        settings = Settings(mode, 0.0, scale=0.9, min_margin=0, divisible_by=1)
        ratios = []
        for size in [128, 64, 32, 16]:
            if mode == "Rectangular":
                w, h = compute_float_rect_dims(size, size, settings, VAE_FACTOR)
                ratio = w / size
            else:
                s = compute_float_hex_size(size, size, settings, VAE_FACTOR)
                ratio = s / (size / 2)
            ratios.append(ratio)

        spread = max(ratios) - min(ratios)
        print(f"{mode} margin=0 ratios: {[f'{r:.6f}' for r in ratios]} spread={spread:.6f}")
        assert spread < 1e-10, f"{mode} float dims not proportional: {ratios}"


def test_margin_not_proportional():
    """min_margin is absolute (latent px), not proportional to size."""
    s = Settings("Rectangular", 0.0, scale=0.9, min_margin=4, divisible_by=1)
    ratios = []
    for size in [128, 64, 32, 16]:
        w, _ = compute_float_rect_dims(size, size, s, VAE_FACTOR)
        ratios.append(w / size)

    print(f"RECT margin=4 ratios by size: {[f'{r:.4f}' for r in ratios]}")
    for i in range(len(ratios) - 1):
        assert ratios[i] > ratios[i + 1], f"ratios should decrease: {ratios}"


# ---------------------------------------------------------------------------
# Subsystem boundary comparison at image resolution (1024x1024)
#
# Wrapping computed at latent resolution (128x128) and upscaled — matching
# the real Conv2d path. Mask now also computed at latent resolution and
# upscaled (after fix). All boundaries must match exactly.
# ---------------------------------------------------------------------------

def _compare_subsystems_at_boundary(label, mode, scale, min_margin, div_by, is_conv2d):
    """Compare all subsystems at 1024x1024 image resolution.

    Subsystems:
    1. Wrapping (calculate_mapping at 128x128 latent, upscaled 8x)
    2. Attention tiling (hex_tiling at patch resolution, upscaled)
    3. Crop mask (create_crop_mask at 1024x1024 — now upscaled from latent)
    4. VAE crop (mask clipped to crop bounding box)

    Wrapping is computed at latent resolution (matching real Conv2d path).
    Mask is also computed at latent resolution and upscaled (after fix).
    They must match exactly — any mismatch is a logic error.
    """
    raw = Settings(mode, 0.0, scale=scale, min_margin=min_margin, divisible_by=div_by)
    settings = raw._resolve_auto(is_conv2d=is_conv2d)

    W_img, H_img = 1024, 1024
    vf = VAE_FACTOR
    W_lat, H_lat = W_img // vf, H_img // vf
    ps = 1 if is_conv2d else 2
    model_type = 'Conv2d' if is_conv2d else 'DiT (patch_size=2)'

    print(f"\n{'='*80}")
    print(f"{label}: {mode} resolved scale={settings.scale} min_margin={settings.min_margin} div_by={div_by}")
    print(f"Image: {W_img}x{H_img}, vae_factor={vf}, {model_type}")
    print(f"{'='*80}")

    # --- Compute subsystems ---

    # 1. Wrapping at LATENT resolution (matching real Conv2d path), upscaled
    wrap_lat = build_wrapping_identity(W_lat, H_lat, settings, vae_factor=vf)
    wrap_np = wrap_lat.numpy() if isinstance(wrap_lat, torch.Tensor) else wrap_lat
    wrap_img = np.repeat(np.repeat(wrap_np, vf, axis=0), vf, axis=1)

    # 2. Attention/tiling at PATCH resolution, upscaled to image
    p_sets, p_vf = ta_mod._make_patch_settings(settings, ps, vf)
    W_p, H_p = W_lat // ps, H_lat // ps
    attn_tile = build_tiling_identity(W_p, H_p, p_sets, p_vf, mode)
    sf = ps * vf
    attn_img = np.repeat(np.repeat(attn_tile, sf, axis=0), sf, axis=1)

    # Attention boundary patches
    attn_bnd = build_boundary_map(H_p, W_p, p_sets, p_vf, mode)
    attn_bnd_img = np.repeat(np.repeat(attn_bnd, sf, axis=0), sf, axis=1)

    # 3. Mask at image resolution (now computed at latent res and upscaled)
    mask = at_mod.create_crop_mask(W_img, H_img, settings, vae_factor=vf)
    mask_img = mask[0, :, :, 0].bool().numpy()

    # 4. Crop
    rmin, rmax, cmin, cmax = at_mod._mask_bounding_box(mask)
    if mode == "Hexagon":
        cr = at_mod.hex_square_crop(rmin, rmax, cmin, cmax, H_img, W_img, div_by)
    else:
        cr = (rmin, rmax, cmin, cmax)
    crop_img = np.zeros((H_img, W_img), dtype=bool)
    crop_img[cr[0]:cr[1]+1, cr[2]:cr[3]+1] = mask_img[cr[0]:cr[1]+1, cr[2]:cr[3]+1]

    # --- Compare wrapping vs mask (must match exactly) ---
    wrap_mask_diff = int(np.sum(wrap_img != mask_img))
    total = W_img * H_img

    wrap_inside = int(wrap_img.sum())
    mask_inside = int(mask_img.sum())
    crop_inside = int(crop_img.sum())
    attn_inside = int(attn_img.sum())

    print(f"\n  Inside counts:")
    print(f"    Wrapping:  {wrap_inside} ({wrap_inside/total*100:.1f}%)")
    print(f"    Mask:      {mask_inside} ({mask_inside/total*100:.1f}%)")
    print(f"    Crop:      {crop_inside} ({crop_inside/total*100:.1f}%)")
    print(f"    Attention: {attn_inside} ({attn_inside/total*100:.1f}%)")
    print(f"    Wrap vs Mask: {wrap_mask_diff} px differ")

    if wrap_mask_diff == 0:
        print(f"    Wrapping == Mask: PERFECT MATCH")
    else:
        print(f"    Wrapping != Mask: MISMATCH")

    # --- Find boundary regions ---
    cy, cx = H_img // 2, W_img // 2

    left_bx_mask = next((x for x in range(W_img) if mask_img[cy, x]), None)
    left_bx_wrap = next((x for x in range(W_img) if wrap_img[cy, x]), None)

    first_y = next((y for y in range(H_img) if mask_img[y].any()), None)
    first_x = next((x for x in range(W_img) if mask_img[first_y, x]), None) if first_y else None

    # --- Visualization ---
    win_h, win_w = 16, 48
    subs = [
        ("1. Wrapping (128x128 latent, 8x8 blocks)", wrap_img),
        (f"2. Attention tiling ({W_p}x{W_p} patches, {sf}x{sf} blocks)", attn_img),
        ("3. Crop mask (latent->image, 8x8 blocks)", mask_img),
        ("4. VAE crop (crop rect bounded)", crop_img),
    ]

    def _show(y0, x0, title):
        y1 = min(H_img, y0 + win_h)
        x1 = min(W_img, x0 + win_w)
        print(f"\n  --- {title} ---")
        for name, grid in subs:
            print(f"\n    {name}:")
            for y in range(y0, y1):
                print(f"      y={y:4d}  " + ''.join(
                    '#' if grid[y, x] else '.' for x in range(x0, x1)))

    def _show_diff(y0, x0, title, a, b, na, nb):
        y1 = min(H_img, y0 + win_h)
        x1 = min(W_img, x0 + win_w)
        print(f"\n    {title}")
        print(f"    (= agree, 1={na}-only, 2={nb}-only)")
        for y in range(y0, y1):
            print(f"      y={y:4d}  " + ''.join(
                '=' if a[y, x] == b[y, x] else ('1' if a[y, x] else '2')
                for x in range(x0, x1)))

    def _show_overlay(y0, x0, title):
        y1 = min(H_img, y0 + win_h)
        x1 = min(W_img, x0 + win_w)
        print(f"\n    {title}")
        print(f"    (# = inside, B = boundary patch, . = outside)")
        for y in range(y0, y1):
            print(f"      y={y:4d}  " + ''.join(
                'B' if attn_bnd_img[y, x] else ('#' if attn_img[y, x] else '.')
                for x in range(x0, x1)))

    # Region 1: Left boundary at center row
    if left_bx_mask is not None:
        y0 = cy - win_h // 2
        x0 = max(0, min(left_bx_mask, left_bx_wrap or left_bx_mask) - 8)
        _show(y0, x0, f"Left boundary at center (y={cy}): mask x={left_bx_mask}, wrap x={left_bx_wrap}")
        _show_diff(y0, x0, "Diff: wrapping vs mask", wrap_img, mask_img, "wrap", "mask")
        _show_overlay(y0, x0, "Attention boundary patches (B = K/V injection)")

    # Region 2: Top-left corner
    if first_y is not None and first_x is not None:
        y0 = max(0, first_y - 2)
        x0 = max(0, first_x - 8)
        _show(y0, x0, f"Top-left corner: first inside at y={first_y}, x={first_x}")
        _show_diff(y0, x0, "Diff: wrapping vs mask", wrap_img, mask_img, "wrap", "mask")

    # Assert: wrapping must match mask exactly (both at same latent resolution)
    assert wrap_mask_diff == 0, (
        f"Wrapping != Mask: {wrap_mask_diff} pixels differ. "
        f"This indicates a logic error in how settings are applied."
    )


def test_subsystem_hex_s1_dit():
    """HEX scale=1.0, min_margin=-1 (auto->0), div_by=1, DiT: working baseline."""
    _compare_subsystems_at_boundary(
        "HEX-DiT-s1.0", "Hexagon", scale=1.0, min_margin=-1, div_by=1, is_conv2d=False)


def test_subsystem_hex_s1_conv2d():
    """HEX scale=1.0, min_margin=-1 (auto->0), div_by=1, Conv2d: working baseline."""
    _compare_subsystems_at_boundary(
        "HEX-Conv2d-s1.0", "Hexagon", scale=1.0, min_margin=-1, div_by=1, is_conv2d=True)


def test_subsystem_hex_s09_dit():
    """HEX scale=0.9, min_margin=-1 (auto->0), div_by=1, DiT: reported broken case."""
    _compare_subsystems_at_boundary(
        "HEX-DiT-s0.9", "Hexagon", scale=0.9, min_margin=-1, div_by=1, is_conv2d=False)


def test_subsystem_hex_s09_conv2d():
    """HEX scale=0.9, min_margin=-1 (auto->0), div_by=1, Conv2d: reported broken case."""
    _compare_subsystems_at_boundary(
        "HEX-Conv2d-s0.9", "Hexagon", scale=0.9, min_margin=-1, div_by=1, is_conv2d=True)


# ---------------------------------------------------------------------------
# Settings resolution tests
# ---------------------------------------------------------------------------


def test_resolve_once_dit_rect_auto():
    """DiT Rect auto: resolved settings produce correct crop matching generation.

    Simulates the full pipeline: settings resolved once in model patcher,
    then used by VAE decode for crop mask. The crop must match the working
    area used during generation (scale=7/8, margin=4).
    """
    # Step 1: User creates settings with auto sentinels
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=1)
    assert not raw.resolved

    # Step 2: Model patcher resolves once (DiT = is_conv2d=False)
    resolved = raw._resolve_auto(is_conv2d=False)
    assert resolved.resolved
    assert resolved.scale == 7 / 8
    assert resolved.min_margin == 4

    # Step 3: VAE decode uses resolved settings directly (no re-resolve)
    vf = VAE_FACTOR
    W_img, H_img = 1024, 1024
    W_lat, H_lat = W_img // vf, H_img // vf

    # Conv2d wrapping at latent resolution uses resolved settings
    wrap_lat = build_wrapping_identity(W_lat, H_lat, resolved, vae_factor=vf)

    # Crop mask at image resolution uses resolved settings
    mask = at_mod.create_crop_mask(W_img, H_img, resolved, vae_factor=vf)
    mask_2d = mask[0, :, :, 0].numpy()
    rmin, rmax, cmin, cmax = at_mod._mask_bounding_box(mask)

    # Wrapping and mask must match exactly
    wrap_np = wrap_lat.numpy() if isinstance(wrap_lat, torch.Tensor) else wrap_lat
    wrap_img = np.repeat(np.repeat(wrap_np, vf, axis=0), vf, axis=1)
    assert np.array_equal(wrap_img, mask_2d > 0), "Wrapping != Mask for DiT Rect auto"

    # Crop dimensions must be sensible (working area, not full image)
    crop_w = cmax - cmin + 1
    crop_h = rmax - rmin + 1
    assert crop_w < W_img, f"Crop should be smaller than image: {crop_w} >= {W_img}"
    assert crop_h < H_img, f"Crop should be smaller than image: {crop_h} >= {H_img}"
    print(f"  DiT Rect auto: crop={crop_w}x{crop_h} (from {W_img}x{H_img})")


def test_resolve_once_dit_rect_auto_div16():
    """DiT Rect auto + div_by=16: crop dimensions divisible by 16."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=16)
    resolved = raw._resolve_auto(is_conv2d=False)

    mask = at_mod.create_crop_mask(1024, 1024, resolved, vae_factor=VAE_FACTOR)
    rmin, rmax, cmin, cmax = at_mod._mask_bounding_box(mask)
    crop_w = cmax - cmin + 1
    crop_h = rmax - rmin + 1
    assert crop_w % 16 == 0, f"crop_w={crop_w} not div by 16"
    assert crop_h % 16 == 0, f"crop_h={crop_h} not div by 16"
    print(f"  DiT Rect auto div16: crop={crop_w}x{crop_h}")


def test_resolve_once_conv2d_rect_auto():
    """Conv2d Rect auto: resolves to scale=1.0, margin=0, no crop needed."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=1)
    resolved = raw._resolve_auto(is_conv2d=True)
    assert resolved.scale == 1.0
    assert resolved.min_margin == 0

    mask = at_mod.create_crop_mask(1024, 1024, resolved, vae_factor=VAE_FACTOR)
    assert mask.sum().item() == 1024 * 1024, "Conv2d Rect auto should have full mask"


def test_resolved_property():
    """Settings.resolved is True when no sentinel values remain."""
    s1 = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=1)
    assert not s1.resolved

    s2 = Settings("Rectangular", 0.0, scale=0.9, min_margin=4, divisible_by=1)
    assert s2.resolved

    s3 = s1._resolve_auto(is_conv2d=False)
    assert s3.resolved
    assert s3.scale != 0.0
    assert s3.min_margin != -1


if __name__ == "__main__":
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
