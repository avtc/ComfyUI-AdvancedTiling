"""
DiT model tiling via latent padding.

Works with any DiT model that supports the model_function_wrapper hook,
including Qwen Image via both standard ComfyUI and raylight FSDP/USP.

On each denoising step, the input latent is padded to a larger size. The
padding region is filled with hex-wrapped content from the opposite edge.
The model processes the padded latent with position IDs computed for the
larger size — padding patches are genuinely adjacent to the boundary in
position space. The output is cropped back to the original size.

This is analogous to how Conv2d tiling wraps padding at every layer, but
applied at the latent level for transformer-based models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d

from .modes import Settings


def _has_conv2d(model: nn.Module) -> bool:
    """Check if the model has any Conv2d layers (UNet vs DiT)."""
    return any(isinstance(m, Conv2d) for m in model.modules())


def create_latent_tiling_wrapper(settings: Settings, padding: int):
    """
    Create a model function wrapper that pads the latent, fills padding
    with hex-wrapped content, runs the model, and crops the output.

    :param settings: Tiling settings
    :param padding: Latent pixels to add on each side
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

        # Pad the latent on all sides
        x_padded = F.pad(x, (padding, padding, padding, padding))

        if is_5d:
            _, _, _, padded_H, padded_W = x_padded.shape
        else:
            _, _, padded_H, padded_W = x_padded.shape

        # Compute hex wrapping mapping for the padded grid
        cache_key = (W, H, padded_W, padded_H, hash(settings))
        if cache_key not in _cache:
            _cache[cache_key] = calculate_mapping(
                (W, H), (padded_W, padded_H), settings
            )
        mapping = _cache[cache_key]

        # Fill padded tensor with wrapped content.
        # Hexagonal positions: identity (source == dest, no-op).
        # Waste/padding positions: copies wrapped content from hex source.
        if is_5d:
            x_padded[:, :, :, mapping[1], mapping[0]] = x_padded[
                :, :, :, mapping[3], mapping[2]
            ]
        else:
            x_padded[:, :, mapping[1], mapping[0]] = x_padded[
                :, :, mapping[3], mapping[2]
            ]

        # Run model on the padded latent
        result = apply_model(x_padded, args["timestep"], **args["c"])

        # Crop back to original size
        p = padding
        if result.ndim == 5:
            result = result[:, :, :, p : p + H, p : p + W]
        else:
            result = result[:, :, p : p + H, p : p + W]

        return result

    return wrapper


def patch_dit_model(model_patcher, settings: Settings, padding: int = 16):
    """
    Apply DiT tiling patch to a ComfyUI ModelPatcher.

    Pads the latent with hex-wrapped content on each denoising step,
    forcing the model to attend to wrapped content at the boundary.

    :param model_patcher: ComfyUI ModelPatcher instance
    :param settings: Tiling settings
    :param padding: Latent pixels to add on each side
    """
    wrapper = create_latent_tiling_wrapper(settings, padding)
    model_patcher.set_model_unet_function_wrapper(wrapper)
