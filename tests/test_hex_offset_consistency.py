"""
Test that wrapping preserves position-within-cell consistency.

For Hexagon mode: every remapped pixel (dest -> source) must land at the same
hex-relative position within its hex cell.  Uses production pixel_to_hex() to
compute fractional hex coordinates, then compares the fractional parts.

For Rectangular mode: every remapped pixel must be displaced by an integer
multiple of the working-area period, producing a seamless periodic tile.

These tests use ONLY production code paths (pixel_to_hex, calculate_mapping,
rect_tiling_at) with no re-implemented math.
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

import numpy as np
from ComfyUI_AdvancedTiling.modes import Settings
from ComfyUI_AdvancedTiling.modes.hex import get_inverse_matrix, axial_round
from ComfyUI_AdvancedTiling.modes.rect import rect_tiling
import ComfyUI_AdvancedTiling.advanced_tiling as at_mod

VAE_FACTOR = 8
IMG_W = 1024
IMG_H = 1024

# 4-neighbor search achieves <0.01 error.  Pre-fix simple rounding had up to
# ~0.5 error.  Threshold 0.05 catches regressions while allowing the inherent
# integer-pixel imprecision.
_HEX_OFFSET_TOL = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(mode, scale, min_margin, div_by, is_conv2d=False):
    s = Settings(mode, 0.0, scale=scale, min_margin=min_margin,
                 divisible_by=div_by, conv2d_attention_wrapping=False)
    return s._resolve_auto(is_conv2d, VAE_FACTOR, 1, IMG_W, IMG_H)


def check_hex_offset_consistency(resolved, res=128, pad=1, tol=_HEX_OFFSET_TOL):
    """For every remapped pixel, source and dest must share the same hex offset.

    Uses production get_inverse_matrix() to compute raw fractional hex
    coordinates (no rounding), then compares the fractional parts of both
    source and destination pixels.  The 4-neighbor search in
    _hex_remap_batch minimises but cannot always eliminate the integer
    quantisation error, so a small tolerance is allowed.
    """
    pw, ph = res + 2 * pad, res + 2 * pad
    hs = resolved.hex_size_at(res, res)
    inv_mat = get_inverse_matrix(resolved.rotation)
    cx = pw // 2
    cy = ph // 2

    mapping = at_mod.calculate_mapping((res, res), (pw, ph), resolved)
    n_mapped = len(mapping[0])
    if n_mapped == 0:
        return

    sx = mapping[0].numpy()
    sy = mapping[1].numpy()
    nx = mapping[2].numpy()
    ny = mapping[3].numpy()

    # Sample up to 2000 pixels to keep the test fast
    if n_mapped > 2000:
        idx = np.linspace(0, n_mapped - 1, 2000, dtype=int)
    else:
        idx = np.arange(n_mapped)

    max_err = 0.0
    n_over = 0
    for i in idx:
        dx, dy = int(sx[i]), int(sy[i])
        snx, sny = int(nx[i]), int(ny[i])

        # Raw fractional hex coords via production inverse matrix
        q1, r1 = (inv_mat @ np.array([[dx - cx], [dy - cy]])).flatten() / hs
        q2, r2 = (inv_mat @ np.array([[snx - cx], [sny - cy]])).flatten() / hs

        # Fractional parts using production axial_round
        rq1, rr1 = axial_round((float(q1), float(r1)))
        rq2, rr2 = axial_round((float(q2), float(r2)))
        frac_dq = abs((q1 - rq1) - (q2 - rq2))
        frac_dr = abs((r1 - rr1) - (r2 - rr2))
        err = max(frac_dq, frac_dr)
        if err > max_err:
            max_err = err
        if err > tol:
            n_over += 1

    assert n_over == 0, (
        f"hex_offset_consistency: {n_over}/{len(idx)} sampled pixels exceed "
        f"tolerance {tol} (max_err={max_err:.6f}, "
        f"hex_size={hs:.2f}, res={res}, pad={pad})"
    )


def check_rect_periodic_consistency(resolved, res=128, pad=1):
    """For every remapped pixel, the displacement must be a near-integer
    multiple of the working-area period.

    Cross-checks calculate_mapping() against the per-pixel production
    rect_tiling() function, using the same coordinate convention
    (work scaled to original resolution, grid over padded size).
    """
    import math

    pw, ph = res + 2 * pad, res + 2 * pad
    # calculate_mapping uses work_at(original_size), not work_at(padded_size)
    work_w, work_h = resolved.work_at(res, res)
    cx, cy = pw / 2.0, ph / 2.0

    mapping = at_mod.calculate_mapping((res, res), (pw, ph), resolved)
    n_mapped = len(mapping[0])
    if n_mapped == 0:
        return

    sx = mapping[0].numpy()
    sy = mapping[1].numpy()
    nx = mapping[2].numpy()
    ny = mapping[3].numpy()

    # Cross-check against production rect_tiling() using same params
    if n_mapped > 2000:
        idx = np.linspace(0, n_mapped - 1, 2000, dtype=int)
    else:
        idx = np.arange(n_mapped)

    n_mismatch = 0
    for i in idx:
        dx, dy = int(sx[i]), int(sy[i])
        expected_nx, expected_ny = rect_tiling(dx, dy, (pw, ph), work_w, work_h)
        if expected_nx != int(nx[i]) or expected_ny != int(ny[i]):
            n_mismatch += 1

    assert n_mismatch == 0, (
        f"rect_periodic_consistency: {n_mismatch}/{len(idx)} sampled pixels "
        f"disagree between calculate_mapping and rect_tiling "
        f"(work=({work_w:.1f},{work_h:.1f}), res={res}, pad={pad})"
    )

    # Verify periodicity: displacement / work should be near-integer
    disp_x = (nx - sx).astype(np.float64)
    disp_y = (ny - sy).astype(np.float64)
    ratio_x = disp_x / work_w
    ratio_y = disp_y / work_h
    frac_x = np.abs(ratio_x - np.round(ratio_x))
    frac_y = np.abs(ratio_y - np.round(ratio_y))
    max_frac = max(frac_x.max(), frac_y.max())

    assert max_frac < 0.01, (
        f"rect_periodic_consistency: displacement not periodic — "
        f"max fractional period error={max_frac:.6f} "
        f"(work=({work_w:.1f},{work_h:.1f}), res={res}, pad={pad})"
    )


# ---------------------------------------------------------------------------
# Hex: sweep margins 0-10, paddings 0-2, and divisible_by values
# ---------------------------------------------------------------------------

def _make_hex_test(margin, div_by, pad):
    def test_fn():
        resolved = _resolve("Hexagon", 1.0, margin, div_by, is_conv2d=True)
        check_hex_offset_consistency(resolved, res=128, pad=pad)
    test_fn.__name__ = f"test_hex_m{margin}_d{div_by}_p{pad}"
    return test_fn


for _m in range(0, 11):
    for _pad in [0, 1, 2]:
        globals()[f"test_hex_m{_m}_d1_p{_pad}".__class__.__name__] = None
        _fn = _make_hex_test(_m, 1, _pad)
        globals()[_fn.__name__] = _fn

for _div in [2, 4, 8, 16]:
    for _pad in [0, 1]:
        _fn = _make_hex_test(0, _div, _pad)
        globals()[_fn.__name__] = _fn
        _fn = _make_hex_test(4, _div, _pad)
        globals()[_fn.__name__] = _fn


# ---------------------------------------------------------------------------
# Rect: sweep margins and divisible_by values
# ---------------------------------------------------------------------------

def _make_rect_test(margin, div_by, pad):
    def test_fn():
        resolved = _resolve("Rectangular", 1.0, margin, div_by, is_conv2d=True)
        check_rect_periodic_consistency(resolved, res=128, pad=pad)
    test_fn.__name__ = f"test_rect_m{margin}_d{div_by}_p{pad}"
    return test_fn


for _m in [0, 4, 8]:
    for _pad in [0, 1, 2]:
        _fn = _make_rect_test(_m, 1, _pad)
        globals()[_fn.__name__] = _fn

for _div in [2, 4, 8, 16]:
    _fn = _make_rect_test(4, _div, 1)
    globals()[_fn.__name__] = _fn


# ---------------------------------------------------------------------------
# Cross-resolution: latent and VAE decoder resolutions
# ---------------------------------------------------------------------------

def test_hex_cross_resolution():
    """Hex offset consistency at UNet resolutions (16+ latent pixels)."""
    for margin in [0, 2, 4, 6]:
        resolved = _resolve("Hexagon", 1.0, margin, 1, is_conv2d=True)
        for res in [128, 64, 32, 16]:
            # At res=16 the hex is only ~8px wide; 4-neighbor search
            # has fewer candidates, so allow higher tolerance.
            tol = 0.1 if res <= 16 else _HEX_OFFSET_TOL
            check_hex_offset_consistency(resolved, res=res, pad=1, tol=tol)


def test_rect_cross_resolution():
    """Rect periodic consistency at all UNet resolutions."""
    for margin in [0, 4, 8]:
        resolved = _resolve("Rectangular", 1.0, margin, 1, is_conv2d=True)
        for res in [128, 64, 32, 16]:
            check_rect_periodic_consistency(resolved, res=res, pad=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
