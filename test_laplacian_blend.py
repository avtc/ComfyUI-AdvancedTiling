"""Tests for Laplacian pyramid blending (pure PyTorch, no ComfyUI dependency)."""

import torch
import sys
import os

# Allow importing from the project root
sys.path.insert(0, os.path.dirname(__file__))

from masked_vae_decode import laplacian_pyramid_blend, _make_smooth_hex_mask


def test_blend_preserves_masked_regions():
    """Hard mask left/right. Far from boundary should match source images."""
    B, H, W, C = 1, 64, 64, 3

    # image_a: all white, image_b: all black
    image_a = torch.ones(B, H, W, C)
    image_b = torch.zeros(B, H, W, C)

    # Hard mask: left half = 1 (keep a), right half = 0 (keep b)
    mask = torch.zeros(1, 1, H, W)
    mask[:, :, :, : W // 2] = 1.0

    result = laplacian_pyramid_blend(image_a, image_b, mask)

    # Far-left pixels should be close to image_a (1.0)
    left_region = result[0, :, :8, :]
    assert left_region.mean().item() > 0.95, (
        f"Left region should be close to image_a (white), got {left_region.mean().item():.4f}"
    )

    # Far-right pixels should be close to image_b (0.0)
    right_region = result[0, :, -8:, :]
    assert right_region.mean().item() < 0.05, (
        f"Right region should be close to image_b (black), got {right_region.mean().item():.4f}"
    )

    print("PASS: test_blend_preserves_masked_regions")


def test_blend_with_smooth_mask():
    """White/black images with smooth gradient mask. Transition should be monotonic."""
    B, H, W, C = 1, 64, 64, 3

    image_a = torch.ones(B, H, W, C)
    image_b = torch.zeros(B, H, W, C)

    # Smooth horizontal gradient mask: 1 on left, 0 on right
    mask = torch.zeros(1, 1, H, W)
    for x in range(W):
        mask[0, 0, :, x] = 1.0 - x / (W - 1)

    result = laplacian_pyramid_blend(image_a, image_b, mask)

    # Average across batch and channels to get (H, W) profile
    profile = result[0].mean(dim=-1).mean(dim=0)  # (W,) column averages

    # In the transition zone (middle 50%), values should be monotonically decreasing
    start = W // 4
    end = 3 * W // 4
    transition = profile[start:end]

    violations = 0
    for i in range(len(transition) - 1):
        if transition[i] < transition[i + 1] - 1e-4:
            violations += 1

    assert violations == 0, (
        f"Transition should be monotonically decreasing, got {violations} violations"
    )

    print("PASS: test_blend_with_smooth_mask")


def test_blend_identity():
    """Blending identical images should return the same image."""
    B, H, W, C = 1, 64, 64, 3

    image = torch.rand(B, H, W, C)
    mask = torch.rand(1, 1, H, W)

    result = laplacian_pyramid_blend(image, image, mask)

    diff = (result - image).abs().max().item()
    assert diff < 0.01, (
        f"Blending identical images should return the same image, max diff = {diff:.6f}"
    )

    print("PASS: test_blend_identity")


def test_smooth_hex_mask_shape_and_range():
    """Smooth hex mask has correct shape and values in [0, 1]."""
    H, W = 128, 128

    # Top half = waste (1.0), bottom half = inside (0.0)
    waste_mask = torch.zeros(1, H, W)
    waste_mask[0, :H // 2, :] = 1.0

    result = _make_smooth_hex_mask(waste_mask, blend_band=16, image_shape=(1, H, W, 3))

    assert result.shape == (1, 1, H, W), f"Expected shape (1, 1, {H}, {W}), got {result.shape}"
    assert result.min().item() >= 0.0, f"Min value should be >= 0, got {result.min().item():.4f}"
    assert result.max().item() <= 1.0, f"Max value should be <= 1, got {result.max().item():.4f}"

    # Bottom half (inside) should be close to 1
    bottom_mean = result[0, 0, H // 2:, :].mean().item()
    assert bottom_mean > 0.9, f"Bottom half (inside) mean should be > 0.9, got {bottom_mean:.4f}"

    # Top half (waste) should be close to 0
    top_mean = result[0, 0, :H // 2, :].mean().item()
    assert top_mean < 0.1, f"Top half (waste) mean should be < 0.1, got {top_mean:.4f}"

    print("PASS: test_smooth_hex_mask_shape_and_range")


def test_smooth_hex_mask_transition():
    """Smooth hex mask transitions from ~0 (waste) to ~1 (inside) at boundary."""
    H, W = 128, 128

    # Top half = waste (1.0), bottom half = inside (0.0)
    waste_mask = torch.zeros(1, H, W)
    waste_mask[0, :H // 2, :] = 1.0

    result = _make_smooth_hex_mask(waste_mask, blend_band=16, image_shape=(1, H, W, 3))

    # Profile: average across batch, channel, and width -> (H,)
    profile = result[0, 0, :, :].mean(dim=-1)  # (H,)

    # Check transition at boundary (row H//2)
    # Value below boundary should be > value above boundary
    above = profile[H // 2 - 1].item()
    below = profile[H // 2].item()
    assert below > above, (
        f"Below boundary ({below:.4f}) should be > above boundary ({above:.4f})"
    )

    # Top region (far from boundary) should be close to 0
    assert profile[:H // 4].mean().item() < 0.05, (
        f"Top region should be close to 0, got {profile[:H // 4].mean().item():.4f}"
    )

    # Bottom region (far from boundary) should be close to 1
    assert profile[3 * H // 4:].mean().item() > 0.95, (
        f"Bottom region should be close to 1, got {profile[3 * H // 4:].mean().item():.4f}"
    )

    print("PASS: test_smooth_hex_mask_transition")


if __name__ == "__main__":
    test_blend_preserves_masked_regions()
    test_blend_with_smooth_mask()
    test_blend_identity()
    test_smooth_hex_mask_shape_and_range()
    test_smooth_hex_mask_transition()
    print("\nAll tests passed!")
