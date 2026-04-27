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
    create_waste_mask,
)

logger = logging.getLogger("ComfyUI-AdvancedTiling")

# Per-direction color for mask visualization and regional prompting
DIRECTION_COLORS = torch.tensor([
    [1.0, 0.0, 0.0],  # E  → Red
    [0.0, 0.0, 1.0],  # NE → Blue
    [0.0, 1.0, 0.0],  # NW → Green
    [1.0, 1.0, 0.0],  # W  → Yellow
    [1.0, 0.5, 0.0],  # SW → Orange
    [0.5, 0.0, 1.0],  # SE → Purple
])
DIRECTION_COLOR_NAMES = ["red", "blue", "green", "yellow", "orange", "purple"]


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
    rotation_steps: int = 0,
) -> torch.Tensor:
    """
    Composite center and neighbor latents into a single latent.

    Places neighbor content in the waste region around the center hex.

    :param center_latent: Center tile latent
    :param neighbor_latents: Dict mapping direction name ("E", "NE", etc.)
                             to neighbor latent
    :param settings: Tiling settings (rotation only)
    :param rotation_steps: Rotate direction mapping by N steps clockwise
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

    n = len(NEIGHBOR_DIRECTIONS)
    direction_to_idx = {name: idx for idx, name in enumerate(NEIGHBOR_DIRECTIONS)}

    total_pasted = 0
    for direction_name, neighbor_latent in neighbor_latents.items():
        dir_idx = direction_to_idx[direction_name]
        target_idx = (dir_idx - rotation_steps) % n
        mask = (neighbor_map == target_idx)  # [H, W] boolean
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
                "feather_sides": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, also feathers the side edges of each mask where no active neighbor is present. Disable to feather only the inner edge (toward hex center).",
                    },
                ),
                "mask_strength_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Minimum denoise strength in the masked area. Non-zero mask values are remapped so the softest feather edge applies at least this much denoising.",
                    },
                ),
                "mask_strength_max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Maximum denoise strength in the masked area. Non-zero mask values are remapped so the strongest area caps at this value.",
                    },
                ),
                "skip_same_neighbors": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, automatically skips inpainting borders where the neighbor image is identical to the center image. Useful when surrounding tiles share the same terrain type.",
                    },
                ),
                "rotate_mapping": (
                    "INT",
                    {
                        "default": 0,
                        "min": -5,
                        "max": 5,
                        "tooltip": "Rotate neighbor image assignments by N steps clockwise. +1: E input → SE region, NE → E. -1: counter-clockwise. Useful when neighbor tiles come from a differently-oriented grid.",
                    },
                ),
                "enable_preview": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable composited preview output (VAE decode of center + neighbor latents). Adds ~0.5-2s per run.",
                    },
                ),
            },
            "optional": {
                f"neighbor_{d}": ("IMAGE", {"tooltip": f"{d} neighbor tile image"})
                for d in NEIGHBOR_DIRECTIONS
            },
        }

    RETURN_TYPES = ("LATENT", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "IMAGE", "IMAGE", "LATENT", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = (
        "LATENT", "MASK",
        *[f"{NEIGHBOR_DIRECTIONS[i]}_{DIRECTION_COLOR_NAMES[i]}"
          for i in range(len(NEIGHBOR_DIRECTIONS))],
        "composited_preview",
        "color_mask",
        "LATENT_paintbrush",
        "paintbrush_preview",
        "WASTE_MASK",
        "WASTE_MASK_IMG",
    )
    FUNCTION = "run"
    CATEGORY = "conditioning"

    def run(self, settings, vae, center_image, inpaint_mode, border_width, feather_radius, feather_sides, mask_strength_min, mask_strength_max, skip_same_neighbors, rotate_mapping=0, enable_preview=False, **kwargs):
        t_start = time.time()

        n = len(NEIGHBOR_DIRECTIONS)
        rotation_steps = rotate_mapping % n

        if rotation_steps:
            logger.info(f"[HexInpaint] Mapping rotated by {rotation_steps} steps CW")

        # 1. VAE-encode center image
        t0 = time.time()
        center_latent = vae.encode(center_image)
        t1 = time.time()
        logger.info(f"[HexInpaint] VAE encode center: {t1-t0:.3f}s, "
                     f"image={center_image.shape}, latent={center_latent.shape}")

        # 2. VAE-encode provided neighbor images
        neighbor_latents = {}
        neighbor_images = {}
        for direction in NEIGHBOR_DIRECTIONS:
            key = f"neighbor_{direction}"
            if key in kwargs and kwargs[key] is not None:
                t_n = time.time()
                neighbor_latents[direction] = vae.encode(kwargs[key])
                neighbor_images[direction] = kwargs[key]
                logger.info(f"[HexInpaint] VAE encode {direction}: "
                             f"{time.time()-t_n:.3f}s, latent={neighbor_latents[direction].shape}")

        logger.info(f"[HexInpaint] Neighbors provided: {list(neighbor_latents.keys())}")

        # 3. Composite latents with rotated direction mapping
        if neighbor_latents:
            t2 = time.time()
            composited = composite_latents(
                center_latent, neighbor_latents, settings, rotation_steps
            )
            logger.info(f"[HexInpaint] Composite: {time.time()-t2:.3f}s")
        else:
            composited = center_latent
            logger.info("[HexInpaint] No neighbor images — skipping compositing")

        # 4. Determine active directions (have non-matching neighbor)
        active_directions = set()
        for i, direction in enumerate(NEIGHBOR_DIRECTIONS):
            if direction in neighbor_images:
                nimg = neighbor_images[direction]
                if nimg.shape != center_image.shape:
                    logger.warning(f"[HexInpaint] {direction} size mismatch "
                                   f"({nimg.shape} vs {center_image.shape}), treating as active")
                    active_directions.add((i - rotation_steps) % n)
                elif skip_same_neighbors and torch.allclose(center_image, nimg, atol=1e-6):
                    logger.info(f"[HexInpaint] Auto-skip {direction}: matches center")
                else:
                    active_directions.add((i - rotation_steps) % n)

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
            W_lat, H_lat, settings, border_width, feather_lat, feather_sides, active_directions
        )

        # Waste-area mask at latent resolution for InpaintVAEDecode
        waste_mask_lat = create_waste_mask(W_lat, H_lat, settings)  # (1, H_lat, W_lat)
        waste_mask_img = create_waste_mask(W_img, H_img, settings)  # (1, H_img, W_img)
        t4 = time.time()
        logger.info(f"[HexInpaint] Latent masks ({W_lat}x{H_lat}): {t4-t3:.3f}s, "
                     f"border={int(border_mask_lat.sum().item())}, feather={feather_lat}px")

        # 6. Remap mask strength range (non-zero values only)
        if mask_strength_min > 0.0 or mask_strength_max < 1.0:
            nonzero = border_mask_lat > 0
            border_mask_lat = torch.where(
                nonzero,
                mask_strength_min + border_mask_lat * (mask_strength_max - mask_strength_min),
                border_mask_lat,
            )
            logger.info(f"[HexInpaint] Strength remap: "
                         f"[0,1] -> [{mask_strength_min:.2f},{mask_strength_max:.2f}]")

        # 7. Build latent dict with noise_mask
        noise_mask = border_mask_lat.unsqueeze(0)  # (1, 1, H_lat, W_lat)
        latent_dict = {"samples": composited, "noise_mask": noise_mask}
        logger.info(f"[HexInpaint] noise_mask shape={noise_mask.shape}, "
                     f"coverage={noise_mask.mean().item():.3f}")

        # 8. Generate masks at image resolution (for output visualization)
        t5 = time.time()
        hex_radius_img = min(W_img, H_img) // 2
        erosion_img = max(1, int(border_width * hex_radius_img))
        feather_img = max(0, round(feather_radius * erosion_img))

        _, full_border_mask, full_neighbor_masks = create_feathered_masks(
            W_img, H_img, settings, border_width, feather_img, feather_sides, active_directions
        )
        logger.info(f"[HexInpaint] Image masks ({W_img}x{H_img}): {time.time()-t5:.3f}s")

        # 9. Remap image-resolution masks to match
        if mask_strength_min > 0.0 or mask_strength_max < 1.0:
            nonzero_img = full_border_mask > 0
            full_border_mask = torch.where(
                nonzero_img,
                mask_strength_min + full_border_mask * (mask_strength_max - mask_strength_min),
                full_border_mask,
            )
            for i in range(len(NEIGHBOR_DIRECTIONS)):
                nz = full_neighbor_masks[i] > 0
                full_neighbor_masks[i] = torch.where(
                    nz,
                    mask_strength_min + full_neighbor_masks[i] * (mask_strength_max - mask_strength_min),
                    full_neighbor_masks[i],
                )

        # 10. Assemble outputs
        outputs = [latent_dict, full_border_mask]
        for i in range(len(NEIGHBOR_DIRECTIONS)):
            outputs.append(full_neighbor_masks[i])

        # output 9: composited preview (pixel-space composite, no VAE round-trip)
        if enable_preview:
            preview_image = center_image.clone()
            neighbor_map_img = _build_neighbor_map(W_img, H_img, settings)
            n_dirs = len(NEIGHBOR_DIRECTIONS)
            dir_idx_map = {name: idx for idx, name in enumerate(NEIGHBOR_DIRECTIONS)}
            for direction_name, neighbor_img in neighbor_images.items():
                dir_idx = dir_idx_map[direction_name]
                target_idx = (dir_idx - rotation_steps) % n_dirs
                waste_mask = (neighbor_map_img == target_idx)
                if waste_mask.any():
                    preview_image[0][waste_mask] = neighbor_img[0][waste_mask]
            outputs.append(preview_image)
        else:
            outputs.append(torch.zeros(1, 1, 1, 3, dtype=torch.float32))

        # output 10: color mask for regional prompting
        color_mask = torch.zeros(H_img, W_img, 3, dtype=torch.float32)
        for i in range(len(NEIGHBOR_DIRECTIONS)):
            mask_i = full_neighbor_masks[i]  # (H_img, W_img)
            color = DIRECTION_COLORS[i]
            for c in range(3):
                color_mask[:, :, c] += mask_i * color[c]
        outputs.append(color_mask.unsqueeze(0))  # (1, H, W, 3)

        # output 11: paintbrush latent (image with solid colors in border regions)
        # Build paintbrush image: original + solid color overlay in border
        solid_color = torch.zeros(H_img, W_img, 3, dtype=torch.float32)
        border_any = torch.zeros(H_img, W_img, dtype=torch.float32)
        for i in range(len(NEIGHBOR_DIRECTIONS)):
            region = (full_neighbor_masks[i] > 0).float()
            border_any = torch.max(border_any, region)
            color = DIRECTION_COLORS[i]
            for c in range(3):
                solid_color[:, :, c] = torch.where(
                    region > 0, color[c], solid_color[:, :, c],
                )
        border_3ch = border_any.unsqueeze(-1)
        paintbrush_img = center_image[0] * (1 - border_3ch) + solid_color * border_3ch
        paintbrush_img = paintbrush_img.unsqueeze(0)  # (1, H, W, 3)

        # Composite neighbor images into waste region of paintbrush preview
        neighbor_map_img = _build_neighbor_map(W_img, H_img, settings)
        direction_to_idx = {name: idx for idx, name in enumerate(NEIGHBOR_DIRECTIONS)}
        for direction_name, neighbor_img in neighbor_images.items():
            dir_idx = direction_to_idx[direction_name]
            target_idx = (dir_idx - rotation_steps) % n
            waste_mask = (neighbor_map_img == target_idx)
            if waste_mask.any():
                paintbrush_img[0][waste_mask] = neighbor_img[0][waste_mask]

        t_pb = time.time()
        paintbrush_latent = vae.encode(paintbrush_img)
        if neighbor_latents:
            paintbrush_composited = composite_latents(
                paintbrush_latent, neighbor_latents, settings, rotation_steps,
            )
        else:
            paintbrush_composited = paintbrush_latent
        paintbrush_latent_dict = {
            "samples": paintbrush_composited,
            "noise_mask": noise_mask,
        }
        outputs.append(paintbrush_latent_dict)
        outputs.append(paintbrush_img)
        logger.info(f"[HexInpaint] Paintbrush latent: {time.time()-t_pb:.3f}s")

        # output 13: waste-area mask (latent resolution)
        outputs.append(waste_mask_lat)
        # output 14: waste-area mask (image resolution)
        outputs.append(waste_mask_img)

        logger.info(f"[HexInpaint] Total: {time.time()-t_start:.3f}s")
        return tuple(outputs)
