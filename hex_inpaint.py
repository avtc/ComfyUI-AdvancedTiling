"""
Hex tile inpainting: composites center + neighbor latents with border masks
for natural edge transitions.

Uses hex geometry directly (not through settings tiling mode) so that:
- Masks and compositing always use hex_tiling
- No Conv2d wrapping is applied (preserves neighbor content in waste region)
"""

import time
import logging
import functools

import torch

from .modes import Settings
from .modes.hex_mask import (
    NEIGHBOR_DIRECTIONS,
    create_feathered_masks,
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
    from .modes.hex import hex_tiling_vectorized
    from .modes.hex_mask import _compute_sector_map

    mapped_x, mapped_y = hex_tiling_vectorized(width, height, settings)

    mapped_x_t = torch.from_numpy(mapped_x)
    mapped_y_t = torch.from_numpy(mapped_y)

    xs = torch.arange(width).expand(height, width)
    ys = torch.arange(height).unsqueeze(1).expand(height, width)
    is_waste = (mapped_x_t != xs) | (mapped_y_t != ys)

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
    generates border masks for natural edge transitions.

    Uses hex geometry directly — does NOT apply Conv2d wrapping or depend
    on the settings tiling mode. The settings only provide hex rotation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "settings": ("ADVANCED_TILING_SETTINGS",),
                "vae": ("VAE",),
                "center_image": ("IMAGE",),
                "inpaint_mode": (
                    ["Masked Denoising"],
                    {
                        "default": "Masked Denoising",
                        "tooltip": "Inpainting mode. 'Masked Denoising' composites neighbor latents into the waste region and applies a noise mask to the border.",
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
                "feather_radius": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Feather radius as fraction of border width. Softens inner and inactive-side edges for smoother transitions. 0 = sharp edges.",
                    },
                ),
                "skip_same_neighbors": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, automatically skips inpainting borders where the neighbor image is identical to the center image. Useful when surrounding tiles share the same terrain type.",
                    },
                ),
            },
            "optional": {
                f"neighbor_{d}": ("IMAGE", {"tooltip": f"{d} neighbor tile image"})
                for d in NEIGHBOR_DIRECTIONS
            },
        }

    RETURN_TYPES = ("LATENT", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = (
        "LATENT", "MASK",
        "neighbor_E", "neighbor_NE", "neighbor_NW",
        "neighbor_W", "neighbor_SW", "neighbor_SE",
    )
    FUNCTION = "run"
    CATEGORY = "conditioning"

    def run(self, settings, vae, center_image, inpaint_mode, border_width, feather_radius, skip_same_neighbors, **kwargs):
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

        # 3. Composite latents
        if neighbor_latents:
            t2 = time.time()
            composited = composite_latents(center_latent, neighbor_latents, settings)
            logger.info(f"[HexInpaint] Composite: {time.time()-t2:.3f}s")
        else:
            composited = center_latent
            logger.info("[HexInpaint] No neighbor images — skipping compositing")

        # 4. Determine active directions (have non-matching neighbor)
        active_directions = set()
        for i, direction in enumerate(NEIGHBOR_DIRECTIONS):
            key = f"neighbor_{direction}"
            if key in kwargs and kwargs[key] is not None:
                if skip_same_neighbors and torch.allclose(center_image, kwargs[key], atol=1e-6):
                    logger.info(f"[HexInpaint] Auto-skip {direction}: matches center")
                else:
                    active_directions.add(i)

        if active_directions:
            logger.info(f"[HexInpaint] Active: "
                         f"{[NEIGHBOR_DIRECTIONS[i] for i in sorted(active_directions)]}")
        else:
            logger.info("[HexInpaint] No active directions")

        # 5. Generate masks at latent resolution with feathering
        t3 = time.time()
        composited_4d, _ = _normalize_latent(composited)
        _, _, H_lat, W_lat = composited_4d.shape
        H_img, W_img = center_image.shape[1], center_image.shape[2]

        hex_radius_lat = min(W_lat, H_lat) // 2
        erosion_lat = max(1, int(border_width * hex_radius_lat))
        feather_lat = max(0, round(feather_radius * erosion_lat))

        _, border_mask_lat, neighbor_masks_lat = create_feathered_masks(
            W_lat, H_lat, settings, border_width, feather_lat, active_directions
        )
        t4 = time.time()
        logger.info(f"[HexInpaint] Latent masks ({W_lat}x{H_lat}): {t4-t3:.3f}s, "
                     f"border={int(border_mask_lat.sum().item())}, feather={feather_lat}px")

        # 6. Build latent dict with noise_mask
        noise_mask = border_mask_lat.unsqueeze(0)  # (1, 1, H_lat, W_lat)
        latent_dict = {"samples": composited, "noise_mask": noise_mask}
        logger.info(f"[HexInpaint] noise_mask shape={noise_mask.shape}, "
                     f"coverage={noise_mask.mean().item():.3f}")

        # 7. Generate masks at image resolution (for output visualization)
        t5 = time.time()
        hex_radius_img = min(W_img, H_img) // 2
        erosion_img = max(1, int(border_width * hex_radius_img))
        feather_img = max(0, round(feather_radius * erosion_img))

        _, full_border_mask, full_neighbor_masks = create_feathered_masks(
            W_img, H_img, settings, border_width, feather_img, active_directions
        )
        t6 = time.time()
        logger.info(f"[HexInpaint] Image masks ({W_img}x{H_img}): {t6-t5:.3f}s")

        outputs = [latent_dict, full_border_mask]
        for i in range(len(NEIGHBOR_DIRECTIONS)):
            outputs.append(full_neighbor_masks[i])

        logger.info(f"[HexInpaint] Total: {time.time()-t_start:.3f}s")
        return tuple(outputs)
