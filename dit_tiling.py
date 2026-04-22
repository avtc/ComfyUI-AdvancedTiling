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
3. After the model predicts noise, the output is blended with a hex-remapped
   version using a feather mask at the hexagonal boundary. This forces the
   sampler to update boundary patches with content consistent with both sides.
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


@functools.cache
def _create_feather_mask(
    width: int, height: int, settings: Settings, blend_width: int,
) -> torch.Tensor:
    """
    Create a feathered mask with smooth transition at the hex boundary.

    Returns a [1, 1, H, W] tensor:
    - ~1.0 far inside the hexagon (keep model prediction)
    - ~0.5 at the boundary (blend)
    - ~0.0 far outside (use remapped prediction)

    :param width: Latent width
    :param height: Latent height
    :param settings: Tiling settings
    :param blend_width: Width of the transition zone in latent pixels
    """
    from .advanced_tiling import create_crop_mask

    mask = create_crop_mask(width, height, settings)  # [1, H, W, 1]
    mask = mask.permute(0, 3, 1, 2)  # [1, 1, H, W]

    if blend_width > 0:
        kernel_size = blend_width * 2 + 1
        sigma = blend_width / 2.0
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
        padding = kernel_size // 2
        mask = F.conv2d(mask, kernel, padding=padding).clamp(0, 1)

    return mask


def create_latent_tiling_wrapper(settings: Settings, blend_width: int = 16):
    """
    Create a model function wrapper that:
    1. Replaces waste content in the latent (for preview)
    2. Runs the model
    3. Applies noise-level enforcement on the output using a feather mask

    :param settings: Tiling settings
    :param blend_width: Width of the feather zone in latent pixels
    """
    from .advanced_tiling import calculate_mapping

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

        # Step 3: Noise-level enforcement — blend output with remapped predictions
        if blend_width > 0:
            mask_key = (W, H, hash(settings), blend_width)
            if mask_key not in _mask_cache:
                _mask_cache[mask_key] = _create_feather_mask(
                    W, H, settings, blend_width
                )

            feather = _mask_cache[mask_key].to(
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

            # Blend: interior keeps model prediction, boundary transitions
            output = feather * output + (1 - feather) * remapped

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
