"""
Hex tile inpainting: composites center + neighbor latents, patches model
for natural border transitions.
"""

import torch
import torch.nn as nn

from .modes import Settings
from .modes.hex_mask import (
    NEIGHBOR_DIRECTIONS,
    create_border_mask,
    create_neighbor_masks,
    _pixel_angle_from_center,
    _angle_to_direction,
)


def _build_neighbor_map(
    width: int, height: int, settings: Settings
) -> torch.Tensor:
    """
    Build a map from pixel position to neighbor direction index.

    For each pixel outside the hex (waste region), assigns a direction
    index (0-5 for E/NE/NW/W/SW/SE). Pixels inside the hex get -1.

    :param width: Latent width
    :param height: Latent height
    :param settings: Tiling settings
    :return: LongTensor of shape (height, width) with values -1 to 5
    """
    from .advanced_tiling import calculate_mapping

    mapping = calculate_mapping((width, height), (width, height), settings)
    neighbor_map = torch.full((height, width), -1, dtype=torch.long)

    for y in range(height):
        for x in range(width):
            mapped_x = mapping[2][y * width + x]
            mapped_y = mapping[3][y * width + x]
            if mapped_x != x or mapped_y != y:
                angle = _pixel_angle_from_center(x, y, width, height)
                direction = _angle_to_direction(angle)
                neighbor_map[y, x] = direction

    return neighbor_map


def composite_latents(
    center_latent: torch.Tensor,
    neighbor_latents: dict[str, torch.Tensor],
    settings: Settings,
) -> torch.Tensor:
    """
    Composite center and neighbor latents into a single latent.

    Places neighbor content in the waste region around the center hex.

    :param center_latent: Center tile latent [B, C, H, W]
    :param neighbor_latents: Dict mapping direction name ("E", "NE", etc.)
                             to neighbor latent [B, C, H, W]
    :param settings: Tiling settings
    :return: Composited latent [B, C, H, W]
    """
    result = center_latent.clone()
    _, _, H, W = result.shape

    neighbor_map = _build_neighbor_map(W, H, settings)
    direction_to_idx = {name: idx for idx, name in enumerate(NEIGHBOR_DIRECTIONS)}

    for direction_name, neighbor_latent in neighbor_latents.items():
        dir_idx = direction_to_idx[direction_name]
        mask = (neighbor_map == dir_idx)  # [H, W] boolean

        if not mask.any():
            continue

        result[:, :, mask] = neighbor_latent[:, :, mask]

    return result


class AdvancedTilingHexInpaint:
    """
    Hex tile inpainting node. Composites center + neighbor latents,
    generates border masks, and patches model for natural edge transitions.
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
        # 1. VAE-encode center image
        center_latent = vae.encode(center_image)

        # 2. VAE-encode provided neighbor images
        neighbor_latents = {}
        for direction in NEIGHBOR_DIRECTIONS:
            key = f"neighbor_{direction}"
            if key in kwargs and kwargs[key] is not None:
                neighbor_latents[direction] = vae.encode(kwargs[key])

        # 3. Composite latents (for Masked Denoising and Hybrid modes)
        use_latent_paste = inpaint_mode in ("Masked Denoising", "Hybrid")

        if use_latent_paste and neighbor_latents:
            composited = composite_latents(center_latent, neighbor_latents, settings)
        else:
            composited = center_latent

        # 4. Generate masks at latent resolution
        _, C, H_lat, W_lat = composited.shape
        H_img, W_img = center_image.shape[1], center_image.shape[2]

        border_mask = create_border_mask(W_lat, H_lat, settings, border_width)
        neighbor_masks = create_neighbor_masks(W_lat, H_lat, settings, border_width)

        # 5. Build latent dict with optional noise_mask
        latent_dict = {"samples": composited}

        if inpaint_mode in ("Masked Denoising", "Hybrid"):
            noise_mask = border_mask.unsqueeze(0)  # (1, 1, H_lat, W_lat)
            latent_dict["noise_mask"] = noise_mask

        # 6. Patch model
        model_copy = model.clone()
        use_attention = inpaint_mode in ("Attention-Only", "Hybrid")

        if use_attention:
            from .hex_neighbor_attention import HexNeighborAttentionPatch

            diff_model = model_copy.model.diffusion_model
            if hasattr(diff_model, 'pe_embedder'):
                patch = HexNeighborAttentionPatch(settings, diff_model.pe_embedder)
                model_copy.set_model_attn1_patch(patch)

        # 7. Prepare outputs
        full_border_mask = create_border_mask(W_img, H_img, settings, border_width)
        full_neighbor_masks = create_neighbor_masks(W_img, H_img, settings, border_width)

        outputs = [
            model_copy,
            latent_dict,
            full_border_mask,
        ]
        for i in range(len(NEIGHBOR_DIRECTIONS)):
            outputs.append(full_neighbor_masks[i])

        return tuple(outputs)
