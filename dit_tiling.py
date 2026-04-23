"""
DiT model tiling via latent content replacement and noise-level enforcement.

Works with any DiT model that supports the model_function_wrapper and
post_input hooks, including Qwen Image.

Two mechanisms work on the input side:
1. Latent content replacement: copies source content to waste positions in
   the input latent on each denoising step (so the KSampler preview shows
   wrapped content).
2. Position ID fix: replaces waste patches' position IDs with their source
   positions so RoPE encodes them at the opposite edge.

On the output side, noise-level enforcement corrects the model's prediction:
3. After the model predicts noise, the output on selected inner hex edges
   (Left, Top-Right, Bottom-Right) is blended with the hex-remapped version.
   blend_amount (0-1) controls blend strength; blend_width controls how far
   into the hex interior the blend zone extends.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn import Conv2d

from .modes import Settings
from .modes.hex import hex_tiling


def _has_conv2d(model: nn.Module) -> bool:
    """Check if the model has any Conv2d layers (UNet vs DiT)."""
    return any(isinstance(m, Conv2d) for m in model.modules())


def _build_waste_mapping(
    h_patches: int,
    w_patches: int,
    settings: Settings,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute waste-to-source index mapping for hex tiling at patch granularity.

    :param h_patches: Number of patch rows
    :param w_patches: Number of patch columns
    :param settings: Tiling settings
    :return: (waste_indices, source_indices) — Long tensors for indexing
    """
    waste = []
    source = []

    for h in range(h_patches):
        for w in range(w_patches):
            new_x, new_y = hex_tiling(
                w, h,
                (w_patches, h_patches),
                (w_patches, h_patches),
                settings,
            )
            if new_x != w or new_y != h:
                waste.append(h * w_patches + w)
                source.append(new_y * w_patches + new_x)

    return (
        torch.tensor(waste, dtype=torch.long),
        torch.tensor(source, dtype=torch.long),
    )


# Axial neighbor direction -> edge name
_EDGE_MAP = {
    (1, 0): "right", (-1, 0): "left",
    (0, 1): "bottom_right", (0, -1): "top_left",
    (1, -1): "top_right", (-1, 1): "bottom_left",
}

# Edges whose inner side gets the flat blend
_BLEND_EDGES = {"left", "top_right", "bottom_right"}


@functools.cache
def _create_blend_zone_mask(
    width: int, height: int, settings: Settings, blend_width: int,
) -> torch.Tensor:
    """
    Create a binary mask for the flat blend zone on Left, Top-Right, and
    Bottom-Right inner edges of the hexagon.

    For each pixel inside the hex within blend_width of the nearest
    Left/TR/BR waste region, the mask is 1.0. Everywhere else is 0.0.

    Returns a [1, 1, H, W] float tensor.

    :param width: Latent width
    :param height: Latent height
    :param settings: Tiling settings
    :param blend_width: How far into the hex interior the blend zone extends
    """
    from .modes.hex import pixel_to_hex, axial_round
    from .advanced_tiling import create_crop_mask

    hex_mask = create_crop_mask(width, height, settings)  # [1, H, W, 1]
    hex_mask = hex_mask.permute(0, 3, 1, 2).squeeze(0).squeeze(0)  # [H, W]

    size = min(width, height) // 2
    cx, cy = width // 2, height // 2

    # Mark waste pixels on Left/TR/BR edges
    waste_mask = torch.zeros(height, width, dtype=torch.float32)
    for y in range(height):
        for x in range(width):
            q, r = pixel_to_hex((x - cx, y - cy), size, settings)
            rounded = axial_round((q, r))
            edge = _EDGE_MAP.get(rounded)
            if edge in _BLEND_EDGES:
                waste_mask[y, x] = 1.0

    # Dilate waste region inward by blend_width using max_pool
    waste_4d = waste_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    current = waste_4d.clone()
    for _ in range(blend_width):
        current = F.max_pool2d(current, 3, stride=1, padding=1).clamp(0, 1)

    # Blend zone: dilated waste that overlaps inside the hex
    zone = (current[0, 0] > 0) & hex_mask.bool()

    result = torch.zeros(1, 1, height, width, dtype=torch.float32)
    result[0, 0][zone] = 1.0
    return result


def create_latent_tiling_wrapper(settings: Settings):
    """
    Create a model function wrapper that:
    1. Replaces waste content in the latent (for preview)
    2. Runs the model
    3. Blends Left/TR/BR inner edges with hex-remapped predictions

    :param settings: Tiling settings (blend_amount and blend_width from node)
    """
    from .advanced_tiling import calculate_mapping

    blend_amount = settings.blend_amount
    blend_width = settings.blend_width
    _mapping_cache = {}
    _mask_cache = {}

    def wrapper(apply_model, args):
        x = args["input"]
        is_5d = x.ndim == 5

        if is_5d:
            _, _, _, H, W = x.shape
        else:
            _, _, H, W = x.shape

        # Step 1: Content replacement in latent (for KSampler preview)
        cache_key = (W, H, hash(settings))
        if cache_key not in _mapping_cache:
            _mapping_cache[cache_key] = calculate_mapping(
                (W, H), (W, H), settings
            )
        mapping = _mapping_cache[cache_key]

        if is_5d:
            x[:, :, :, mapping[1], mapping[0]] = x[:, :, :, mapping[3], mapping[2]]
        else:
            x[:, :, mapping[1], mapping[0]] = x[:, :, mapping[3], mapping[2]]

        # Step 2: Model prediction
        output = apply_model(args["input"], args["timestep"], **args["c"])

        # Step 3: Flat blend on Left/TR/BR inner edges
        if blend_amount > 0 and blend_width > 0:
            mask_key = (W, H, hash(settings), blend_width)
            if mask_key not in _mask_cache:
                _mask_cache[mask_key] = _create_blend_zone_mask(
                    W, H, settings, blend_width
                )

            zone = _mask_cache[mask_key].to(
                device=output.device, dtype=output.dtype
            )

            # Remapped output: each pixel gets prediction from its hex source
            remapped = torch.empty_like(output)
            if is_5d:
                remapped[:, :, :, mapping[1], mapping[0]] = output[
                    :, :, :, mapping[3], mapping[2]
                ]
            else:
                remapped[:, :, mapping[1], mapping[0]] = output[
                    :, :, mapping[3], mapping[2]
                ]

            # Flat blend: blend_amount * zone acts as alpha
            weight = blend_amount * zone
            output = (1 - weight) * output + weight * remapped

        return output

    return wrapper


def create_img_ids_patch(settings: Settings):
    """
    Create a post_input hook that replaces waste patches' position IDs
    with their source positions, so RoPE encodes them at the opposite edge.

    :param settings: Tiling settings
    :return: Post_input hook function
    """
    _cache = {}

    def post_input(data: dict) -> dict:
        img_ids = data["img_ids"]

        h_positions = img_ids[0, :, 1].unique().sort()[0]
        w_positions = img_ids[0, :, 2].unique().sort()[0]
        h_patches = h_positions.shape[0]
        w_patches = w_positions.shape[0]

        cache_key = (h_patches, w_patches, hash(settings))
        if cache_key not in _cache:
            _cache[cache_key] = _build_waste_mapping(
                h_patches, w_patches, settings,
            )

        waste_idx, source_idx = _cache[cache_key]
        img_ids[:, waste_idx] = img_ids[:, source_idx]

        return data

    return post_input


def patch_dit_model(model_patcher, settings: Settings):
    """
    Apply DiT tiling patch to a ComfyUI ModelPatcher.

    Uses content replacement + position ID fix on the input side,
    and noise-level enforcement on the output side.

    :param model_patcher: ComfyUI ModelPatcher instance
    :param settings: Tiling settings
    """
    wrapper = create_latent_tiling_wrapper(settings)
    model_patcher.set_model_unet_function_wrapper(wrapper)

    img_ids_patch = create_img_ids_patch(settings)
    model_patcher.set_model_post_input_patch(img_ids_patch)
