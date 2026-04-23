"""
DiT model tiling via toroidal attention.

Uses attn1_patch to inject wrapped-neighbor K/V entries into every attention
layer, making boundary patches structurally "see" opposite-edge content as
spatially adjacent. Works with any DiT model that supports the attn1_patch
hook, including Qwen Image.

Optional features (enabled via Settings):
- Latent content wrapping: copies source content to waste positions in the
  input latent on each denoising step (for KSampler preview).
- Position ID fix: replaces waste patches' position IDs with their source
  positions so RoPE encodes them at the opposite edge.
- Output blend: blends Left/TR/BR inner hex edges with hex-remapped predictions.
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d

from .modes import Settings
from .modes.hex import hex_tiling, pixel_to_hex, axial_round


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

    Returns a [1, 1, H, W] float tensor.
    """
    from .advanced_tiling import create_crop_mask

    hex_mask = create_crop_mask(width, height, settings)
    hex_mask = hex_mask.permute(0, 3, 1, 2).squeeze(0).squeeze(0)  # [H, W]

    size = min(width, height) // 2
    cx, cy = width // 2, height // 2

    waste_mask = torch.zeros(height, width, dtype=torch.float32)
    for y in range(height):
        for x in range(width):
            q, r = pixel_to_hex((x - cx, y - cy), size, settings)
            rounded = axial_round((q, r))
            edge = _EDGE_MAP.get(rounded)
            if edge in _BLEND_EDGES:
                waste_mask[y, x] = 1.0

    # Dilate waste region inward by blend_width
    waste_4d = waste_mask.unsqueeze(0).unsqueeze(0)
    current = waste_4d.clone()
    for _ in range(blend_width):
        current = F.max_pool2d(current, 3, stride=1, padding=1).clamp(0, 1)

    # Blend zone: dilated waste that overlaps inside the hex
    zone = (current[0, 0] > 0) & hex_mask.bool()

    result = torch.zeros(1, 1, height, width, dtype=torch.float32)
    result[0, 0][zone] = 1.0
    return result


def create_model_wrapper(settings: Settings):
    """
    Create a model function wrapper that handles:
    1. Content replacement in the latent (for KSampler preview)
    2. Output blend on Left/TR/BR inner hex edges

    :param settings: Tiling settings
    """
    from .advanced_tiling import calculate_mapping

    _mapping_cache = {}
    _mask_cache = {}

    do_content_replace = settings.latent_wrapping
    do_blend = settings.blend_amount > 0 and settings.blend_width > 0

    def wrapper(apply_model, args):
        x = args["input"]
        is_5d = x.ndim == 5

        if is_5d:
            _, _, _, H, W = x.shape
        else:
            _, _, H, W = x.shape

        # Compute mapping (needed for both content replacement and blend)
        cache_key = (W, H, hash(settings))
        if cache_key not in _mapping_cache:
            _mapping_cache[cache_key] = calculate_mapping(
                (W, H), (W, H), settings
            )
        mapping = _mapping_cache[cache_key]

        # Step 1: Content replacement in latent (for KSampler preview)
        if do_content_replace:
            if is_5d:
                x[:, :, :, mapping[1], mapping[0]] = x[:, :, :, mapping[3], mapping[2]]
            else:
                x[:, :, mapping[1], mapping[0]] = x[:, :, mapping[3], mapping[2]]

        # Step 2: Model prediction
        output = apply_model(args["input"], args["timestep"], **args["c"])

        # Step 3: Flat blend on Left/TR/BR inner edges
        if do_blend:
            mask_key = (W, H, hash(settings), settings.blend_width)
            if mask_key not in _mask_cache:
                _mask_cache[mask_key] = _create_blend_zone_mask(
                    W, H, settings, settings.blend_width
                )

            zone = _mask_cache[mask_key].to(
                device=output.device, dtype=output.dtype
            )

            remapped = torch.empty_like(output)
            if is_5d:
                remapped[:, :, :, mapping[1], mapping[0]] = output[
                    :, :, :, mapping[3], mapping[2]
                ]
            else:
                remapped[:, :, mapping[1], mapping[0]] = output[
                    :, :, mapping[3], mapping[2]
                ]

            weight = settings.blend_amount * zone
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
    Apply hex tiling to a DiT model.

    Optionally applies toroidal attention, latent wrapping with blend,
    and position ID fix based on settings.

    :param model_patcher: ComfyUI ModelPatcher instance
    :param settings: Tiling settings
    """
    diff_model = model_patcher.model.diffusion_model

    # Toroidal attention (optional)
    if settings.toroidal_attention:
        from .toroidal_attention import HexToroidalAttentionPatch

        if not hasattr(diff_model, 'pe_embedder'):
            raise ValueError(
                "Model does not have pe_embedder. "
                "Toroidal attention requires a model with RoPE position embeddings."
            )

        patch = HexToroidalAttentionPatch(settings, diff_model.pe_embedder)
        model_patcher.set_model_attn1_patch(patch)

    # Model wrapper: latent content wrapping + output blend
    needs_wrapper = (
        settings.latent_wrapping
        or (settings.blend_amount > 0 and settings.blend_width > 0)
    )
    if needs_wrapper:
        wrapper = create_model_wrapper(settings)
        model_patcher.set_model_unet_function_wrapper(wrapper)

    # Position ID fix (optional, waste patches get source position IDs)
    if settings.position_fix:
        img_ids_patch = create_img_ids_patch(settings)
        model_patcher.set_model_post_input_patch(img_ids_patch)
