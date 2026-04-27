# Laplacian Pyramid Blend for Hex Tile Seam Fix

**Date**: 2026-04-27
**Branch**: feature/inpaint-transitions-of-hexagon-tiles

## Problem

After VAE decode, hex tile boundaries show visible seams (~16px bleed zone). The current `inject_waste` per-stage compositing uses a hard mask, producing a sharp transition at the hex boundary. The seam is especially visible when adjacent tiles have different terrain types (e.g., dark forest meeting light water).

## Solution

Add Laplacian pyramid blending as a post-processing step in `InpaintVAEDecode`. This blends the decoded inpainted image with the `original_image` reference at the hex boundary using a smooth mask at multiple frequency scales.

**Why Laplacian pyramid**: Low-frequency components (color/tone) blend over a wider band, eliminating color mismatch. High-frequency components (detail) blend over a narrower band, preserving sharp features. This produces seamless transitions without ghosting.

## Approach

**Blend between two images**:
- **Image A**: Decoded inpainted latent (correct hex interior, may have artifacts at boundary)
- **Image B**: `original_image` (correct neighbor content in waste area, already at full resolution)

**Smooth hex mask**: Derived from `waste_mask`, inverted (1.0 = inside hex, 0.0 = waste), Gaussian-blurred with sigma proportional to `blend_band`. Creates a smooth transition zone at the hex boundary.

**Algorithm**:
1. Build Gaussian pyramids for both images and the mask (downsample by 2 at each level)
2. Build Laplacian pyramids: `L[i] = G[i] - upsample(G[i+1])`
3. At each level: `L_blend[i] = L_a[i] * G_mask[i] + L_b[i] * (1 - G_mask[i])`
4. Reconstruct from coarsest to finest: `result[i] = upsample(result[i+1]) + L_blend[i]`

**Result**: Waste region stays as `original_image` content (mask=0). Deep inside hex stays as decoded inpainted content (mask=1). Only the transition band at the hex boundary gets multi-scale blended.

## Node Interface Changes

### `InpaintVAEDecode` (`masked_vae_decode.py`)

**Removed inputs**:
- `source_samples` — no longer needed, `original_image` is always required

**Changed inputs**:
- `original_image` — moved from optional to required

**New inputs**:
- `laplacian_blend`: BOOLEAN (default False) — enable post-processing
- `blend_band`: INT (default 16, range 4-64) — transition band width in pixels

**Execution flow**:

| `inject_waste` | `laplacian_blend` | Behavior |
|---|---|---|
| True | False | Per-stage compositing using `original_image` as reference |
| True | True | Per-stage compositing + Laplacian post-smooth |
| False | True | Direct decode + Laplacian blend |
| False | False | Plain VAE decode |

## New Functions

### `laplacian_pyramid_blend(image_a, image_b, mask, levels)`

Core algorithm. Takes decoded image and original_image (both B,H,W,C), smooth mask (1,1,H,W). Returns blended image (B,H,W,C).

### `_make_smooth_hex_mask(waste_mask, blend_band, image_shape)`

Inverts waste_mask (1=inside, 0=waste), applies Gaussian blur with `sigma = blend_band / 3`. Returns (1,1,H,W) float tensor.

## Files Modified

1. **`masked_vae_decode.py`** — all changes in this file:
   - Remove `source_samples` input
   - Move `original_image` to required
   - Add `laplacian_blend` and `blend_band` inputs
   - Add `laplacian_pyramid_blend()` function
   - Add `_make_smooth_hex_mask()` function
   - Update `decode()` to apply Laplacian blend after VAE decode
