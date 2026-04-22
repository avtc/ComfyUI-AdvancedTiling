"""
DiT model tiling via latent content replacement.

Works with any DiT model that supports the post_input transformer_options hook,
including Qwen Image via both standard ComfyUI and raylight FSDP/USP.

On each denoising step, the input latent's waste region (outside the hexagonal
tile boundary) is overwritten with content from the opposite edge via hex
coordinate remapping. This ensures the model sees wrapped content, the VAE
output shows wrapped content (not noise), and the KSampler preview reflects it.

A post_input hook also replaces the position IDs (img_ids) for waste patches
to match their source positions, so RoPE encodes them at the opposite edge.
"""

import torch
import torch.nn as nn
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


def create_latent_tiling_wrapper(settings: Settings):
    """
    Create a model function wrapper that copies source content to waste
    positions in the input latent on each denoising step.

    Modifies the latent in-place so the KSampler preview shows wrapped
    content at waste positions.

    :param settings: Tiling settings
    :return: Wrapper function for set_model_unet_function_wrapper
    """
    from .advanced_tiling import calculate_mapping

    _cache = {}

    def wrapper(apply_model, args):
        x = args["input"]
        is_5d = x.ndim == 5

        if is_5d:
            _, _, _, H, W = x.shape
        else:
            _, _, H, W = x.shape

        cache_key = (W, H, hash(settings))
        if cache_key not in _cache:
            _cache[cache_key] = calculate_mapping(
                (W, H), (W, H), settings
            )

        mapping = _cache[cache_key]

        # Copy source content to waste positions in-place
        if is_5d:
            x[:, :, :, mapping[1], mapping[0]] = x[:, :, :, mapping[3], mapping[2]]
        else:
            x[:, :, mapping[1], mapping[0]] = x[:, :, mapping[3], mapping[2]]

        return apply_model(args["input"], args["timestep"], **args["c"])

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
        transformer_options = data["transformer_options"]

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

        # Replace waste patches' position IDs to match source positions
        img_ids[:, waste_idx] = img_ids[:, source_idx]

        return data

    return post_input


def patch_dit_model(model_patcher, settings: Settings):
    """
    Apply DiT tiling patch to a ComfyUI ModelPatcher.

    Uses a model function wrapper to copy source content to waste positions
    in the latent on each denoising step, and a post_input hook to fix
    position IDs for waste patches.

    :param model_patcher: ComfyUI ModelPatcher instance
    :param settings: Tiling settings
    """
    wrapper = create_latent_tiling_wrapper(settings)
    model_patcher.set_model_unet_function_wrapper(wrapper)

    img_ids_patch = create_img_ids_patch(settings)
    model_patcher.set_model_post_input_patch(img_ids_patch)
