"""
Hex tile inpainting: composites center + neighbor latents, patches model
for natural border transitions.

Uses hex geometry directly (not through settings tiling mode) so that:
- Masks and compositing always use hex_tiling
- No Conv2d wrapping is applied (preserves neighbor content in waste region)
- Attention injects K/V from waste region (neighbor content) at boundaries
"""

import time
import logging
import functools

import torch

from .modes import Settings
from .modes.hex import hex_tiling
from .modes.hex_mask import (
    NEIGHBOR_DIRECTIONS,
    create_masks,
)

logger = logging.getLogger("ComfyUI-AdvancedTiling")


def _normalize_latent(latent: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """
    Normalize latent to 4D (B, C, H, W) by squeezing singleton temporal dims.

    Video/3D VAEs (e.g. wan) return 5D tensors (1, C, T, H, W).
    This squeezes singleton dims between channel and spatial to get 4D.

    :return: (4D tensor, original shape for restoration)
    """
    original_shape = latent.shape
    x = latent
    while x.dim() > 4:
        squeezed = False
        for d in range(1, x.dim() - 2):
            if x.shape[d] == 1:
                x = x.squeeze(d)
                squeezed = True
                break
        if not squeezed:
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
            break
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x, original_shape


@functools.cache
def _build_neighbor_map(
    width: int, height: int, settings: Settings
) -> torch.Tensor:
    """
    Build a map from pixel position to neighbor direction index.

    Uses hex_tiling directly (not settings.tiling_fn) so the hex geometry
    is always correct regardless of the settings mode.

    For each pixel outside the hex (waste region), assigns a direction
    index (0-5 for E/NE/NW/W/SW/SE). Pixels inside the hex get -1.

    :return: LongTensor of shape (height, width) with values -1 to 5
    """
    from .modes.hex_mask import _compute_sector_map

    # Vectorized waste detection using hex_tiling directly
    mapped_x = torch.zeros((height, width), dtype=torch.long)
    mapped_y = torch.zeros((height, width), dtype=torch.long)
    for y in range(height):
        for x in range(width):
            nx, ny = hex_tiling(x, y, (width, height), (width, height), settings)
            mapped_x[y, x] = nx
            mapped_y[y, x] = ny

    xs = torch.arange(width).expand(height, width)
    ys = torch.arange(height).unsqueeze(1).expand(height, width)
    is_waste = (mapped_x != xs) | (mapped_y != ys)

    sectors = _compute_sector_map(width, height)

    neighbor_map = torch.full((height, width), -1, dtype=torch.long)
    neighbor_map[is_waste] = sectors[is_waste]
    return neighbor_map


def composite_latents(
    center_latent: torch.Tensor,
    neighbor_latents: dict[str, torch.Tensor],
    settings: Settings,
) -> torch.Tensor:
    """
    Composite center and neighbor latents into a single latent.

    Places neighbor content in the waste region around the center hex.

    :param center_latent: Center tile latent
    :param neighbor_latents: Dict mapping direction name ("E", "NE", etc.)
                             to neighbor latent
    :param settings: Tiling settings (rotation only)
    :return: Composited latent (same shape as input)
    """
    t0 = time.time()
    result, original_shape = _normalize_latent(center_latent)
    B, C, H, W = result.shape

    t1 = time.time()
    neighbor_map = _build_neighbor_map(W, H, settings)
    t2 = time.time()

    total_waste = (neighbor_map >= 0).sum().item()
    logger.info(f"composite_latents: waste={total_waste}/{H*W}, "
                f"shape=({B},{C},{H},{W}), "
                f"normalize={t1-t0:.3f}s, neighbor_map={t2-t1:.3f}s")

    direction_to_idx = {name: idx for idx, name in enumerate(NEIGHBOR_DIRECTIONS)}

    total_pasted = 0
    for direction_name, neighbor_latent in neighbor_latents.items():
        dir_idx = direction_to_idx[direction_name]
        mask = (neighbor_map == dir_idx)  # [H, W] boolean
        count = mask.sum().item()

        if count == 0:
            logger.info(f"  {direction_name}: no waste pixels found")
            continue

        neighbor_4d, _ = _normalize_latent(neighbor_latent)
        result[:, :, mask] = neighbor_4d[:, :, mask]
        total_pasted += count
        logger.info(f"  {direction_name}: pasted {count} pixels")

    logger.info(f"  total pasted: {total_pasted}/{total_waste} waste pixels")

    if len(original_shape) > 4:
        result = result.reshape(original_shape)
    return result


class AdvancedTilingHexInpaint:
    """
    Hex tile inpainting node. Composites center + neighbor latents,
    generates border masks, and patches model for natural edge transitions.

    Uses hex geometry directly — does NOT apply Conv2d wrapping or depend
    on the settings tiling mode. The settings only provide hex rotation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "settings": ("ADVANCED_TILING_SETTINGS",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "center_image": ("IMAGE",),
                "inpaint_mode": (
                    ["Masked Denoising", "Attention-Only", "Hybrid"],
                    {
                        "default": "Hybrid",
                        "tooltip": "Inpainting mode. 'Hybrid' combines latent pasting + attention. 'Masked Denoising' uses latent paste + noise mask. 'Attention-Only' uses just attention context.",
                    },
                ),
                "border_width": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.05,
                        "max": 0.45,
                        "step": 0.01,
                        "tooltip": "Width of the inpainting border as fraction of hex radius.",
                    },
                ),
            },
            "optional": {
                f"neighbor_{d}": ("IMAGE", {"tooltip": f"{d} neighbor tile image"})
                for d in NEIGHBOR_DIRECTIONS
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = (
        "MODEL", "LATENT", "MASK",
        "neighbor_E", "neighbor_NE", "neighbor_NW",
        "neighbor_W", "neighbor_SW", "neighbor_SE",
    )
    FUNCTION = "run"
    CATEGORY = "conditioning"

    def run(self, settings, model, vae, center_image, inpaint_mode, border_width, **kwargs):
        t_start = time.time()

        # 1. VAE-encode center image
        t0 = time.time()
        center_latent = vae.encode(center_image)
        t1 = time.time()
        logger.info(f"[HexInpaint] VAE encode center: {t1-t0:.3f}s, "
                     f"image={center_image.shape}, latent={center_latent.shape}")

        # 2. VAE-encode provided neighbor images
        neighbor_latents = {}
        for direction in NEIGHBOR_DIRECTIONS:
            key = f"neighbor_{direction}"
            if key in kwargs and kwargs[key] is not None:
                t_n = time.time()
                neighbor_latents[direction] = vae.encode(kwargs[key])
                logger.info(f"[HexInpaint] VAE encode {direction}: "
                             f"{time.time()-t_n:.3f}s, latent={neighbor_latents[direction].shape}")

        logger.info(f"[HexInpaint] Neighbors provided: {list(neighbor_latents.keys())}")

        # 3. Composite latents (for Masked Denoising and Hybrid modes)
        use_latent_paste = inpaint_mode in ("Masked Denoising", "Hybrid")

        if use_latent_paste and neighbor_latents:
            t2 = time.time()
            composited = composite_latents(center_latent, neighbor_latents, settings)
            logger.info(f"[HexInpaint] Composite: {time.time()-t2:.3f}s")
        else:
            composited = center_latent
            if use_latent_paste:
                logger.info("[HexInpaint] No neighbor images — skipping compositing")
            else:
                logger.info(f"[HexInpaint] Mode '{inpaint_mode}' — skipping compositing")

        # 4. Generate masks at latent resolution (single pass)
        t3 = time.time()
        composited_4d, _ = _normalize_latent(composited)
        _, _, H_lat, W_lat = composited_4d.shape
        H_img, W_img = center_image.shape[1], center_image.shape[2]

        _, border_mask_lat, neighbor_masks_lat = create_masks(
            W_lat, H_lat, settings, border_width
        )
        t4 = time.time()
        logger.info(f"[HexInpaint] Latent masks ({W_lat}x{H_lat}): {t4-t3:.3f}s, "
                     f"border_pixels={int(border_mask_lat.sum().item())}")

        # 5. Build latent dict with optional noise_mask
        latent_dict = {"samples": composited}

        if inpaint_mode in ("Masked Denoising", "Hybrid"):
            noise_mask = border_mask_lat.unsqueeze(0)  # (1, 1, H_lat, W_lat)
            latent_dict["noise_mask"] = noise_mask
            logger.info(f"[HexInpaint] noise_mask shape={noise_mask.shape}, "
                         f"coverage={noise_mask.mean().item():.3f}")

        # 6. Patch model — inject K/V from waste region (neighbor content)
        #    at boundary positions. No Conv2d wrapping.
        model_copy = model.clone()
        use_attention = inpaint_mode in ("Attention-Only", "Hybrid")

        if use_attention:
            from .hex_neighbor_attention import HexNeighborAttentionPatch

            diff_model = model_copy.model.diffusion_model
            if hasattr(diff_model, 'pe_embedder'):
                patch = HexNeighborAttentionPatch(settings, diff_model.pe_embedder)
                model_copy.set_model_attn1_patch(patch)
                logger.info("[HexInpaint] Neighbor attention patch applied")
            else:
                logger.info("[HexInpaint] No pe_embedder — attention patch skipped")

        # 7. Generate masks at image resolution (for output visualization)
        t5 = time.time()
        _, full_border_mask, full_neighbor_masks = create_masks(
            W_img, H_img, settings, border_width
        )
        t6 = time.time()
        logger.info(f"[HexInpaint] Image masks ({W_img}x{H_img}): {t6-t5:.3f}s")

        outputs = [
            model_copy,
            latent_dict,
            full_border_mask,
        ]
        for i in range(len(NEIGHBOR_DIRECTIONS)):
            outputs.append(full_neighbor_masks[i])

        logger.info(f"[HexInpaint] Total: {time.time()-t_start:.3f}s")
        return tuple(outputs)
