"""
Main advanced tiling implementation
"""

from typing import Optional
import functools
import copy

import torch
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from .modes import modes, Settings
from .dit_tiling import patch_dit_model, _has_conv2d, _create_content_wrapper


@functools.cache
def calculate_mapping(
    original_size: tuple[int, int], padded_size: tuple[int, int], settings: Settings
):
    """
    Calculate mapping for pixels outside of the mask

    :param original_size: Original size of the image
    :param padded_size: Padded size of the image
    :param settings: Tiling settings
    :return: Mapping of pixels
    """

    mapping = []
    for y in range(padded_size[1]):
        for x in range(padded_size[0]):
            (new_x, new_y) = settings.tiling_fn(
                x, y, original_size, padded_size, settings
            )
            mapping.append([x, y, new_x, new_y])
    return list(zip(*mapping))


@functools.cache
def create_crop_mask(width: int, height: int, settings: Settings):
    """
    Crop image based on tiling settings

    :param image: Image to crop
    :param settings: Tiling settings
    :return: Cropped image
    """

    mask = torch.zeros((1, height, width, 1), dtype=torch.float32)
    for y in range(height):
        for x in range(width):
            # Calculate new coordinates
            (new_x, new_y) = settings.tiling_fn(
                x, y, (width, height), (width, height), settings
            )

            # If coordinates match, it means we are in the mask
            if new_x == x and new_y == y:
                mask[:, y, x] = 1
    return mask


def patch_model(model, settings: Settings):
    """
    Patch model to perform tiling - in place!

    :param model: Model to patch
    :param settings: Tiling settings
    """

    # Patch all Conv2d layers
    for layer in [layer for layer in model.modules() if isinstance(layer, Conv2d)]:
        # pylint: disable=protected-access, no-value-for-parameter
        layer._conv_forward = tiling_conv.__get__(layer, Conv2d)
        layer.tiling_settings = settings
    return


def tiling_conv(self, input_tensor: Tensor, weight: Tensor, bias: Optional[Tensor]):
    """
    Patched Conv2D forward function for tiling

    :param input_tensor: Input tensor
    :param weight: Weight tensor
    :param bias: Bias tensor
    :return: Convolution result
    """

    # Pad input tensor
    padded = F.pad(
        input_tensor,
        # pylint: disable=protected-access
        self._reversed_padding_repeated_twice,
    )
    # Calculate mapping
    mapping = calculate_mapping(
        (input_tensor.shape[-1], input_tensor.shape[-2]),
        (padded.shape[-1], padded.shape[-2]),
        self.tiling_settings,
    )
    # Apply tiling
    padded[:, :, mapping[1], mapping[0]] = padded[:, :, mapping[3], mapping[2]]
    # Perform convolution
    # pylint: disable=not-callable
    return F.conv2d(
        padded, weight, bias, self.stride, _pair(0), self.dilation, self.groups
    )


class AdvancedTilingSettings:
    """
    Tiling settings node that outputs tiling settings for other nodes
    """

    # pylint: disable=invalid-name

    @classmethod
    def INPUT_TYPES(cls):
        """
        Input types for the node
        """

        return {
            "required": {
                "mode": (list(modes.keys()), {
                    "tooltip": "Tiling mode. 'None' disables tiling, 'Hexagon' wraps edges in a hexagonal pattern, 'Rectangular' wraps right→left and bottom→top.",
                }),
                "rotation": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 360.0, "step": 0.01,
                        "tooltip": "Rotation angle in degrees for the tiling pattern.",
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": "Working area scale relative to latent size. 0 = auto (Hexagon: 1.0, Rectangular: ~0.87 matching hex width ratio). Lower values create larger margins for better wrapping at the cost of output size.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("ADVANCED_TILING_SETTINGS",)
    RETURN_NAMES = ("SETTINGS",)
    FUNCTION = "run"

    def run(self, mode, rotation, scale):
        """
        Creates tiling settings from node inputs
        """

        import math

        if scale == 0.0:
            scale = 1.0 if mode == "Hexagon" else math.sqrt(3) / 2

        settings = Settings(mode, rotation, scale)

        return (settings,)


class AdvancedTiling:
    """
    Patches model to perform tiling - supports both UNet (Conv2d) and DiT models
    """

    # pylint: disable=invalid-name

    @classmethod
    def INPUT_TYPES(cls):
        """
        Input types for the node
        """

        return {
            "required": {
                "settings": ("ADVANCED_TILING_SETTINGS",),
                "model": ("MODEL",),
            },
        }

    CATEGORY = "conditioning"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"

    def run(self, settings, model):
        """
        Does the actual patching of the model
        """

        model_copy = model.clone()

        if _has_conv2d(model_copy.model.diffusion_model):
            patch_model(model_copy.model, settings)

            if settings.mode == "Rectangular" and settings.scale < 1.0:
                wrapper = _create_content_wrapper(settings)
                model_copy.set_model_unet_function_wrapper(wrapper)
        else:
            patch_dit_model(model_copy, settings)

        return (model_copy,)


class AdvancedTilingVAEDecode:
    """
    Decode latents with tiling-aware VAE and optionally crop to working area.
    """

    # pylint: disable=invalid-name

    @classmethod
    def INPUT_TYPES(cls):
        """
        Input types for the node
        """

        return {
            "required": {
                "settings": ("ADVANCED_TILING_SETTINGS",),
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "crop": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "latent"

    def run(self, settings, samples, vae, crop):
        """
        Decode latents to image with tiling
        Optionally crop the image based on tiling settings

        For Hexagon mode: applies alpha mask for hex-shaped cropping.
        For Rectangular mode with scale < 1.0: crops the latent to the centered
        working rectangle before VAE decoding, so the VAE's Conv2d wrapping
        operates at the working rectangle boundary.
        """

        from .dit_tiling import _compute_working_size

        vae_copy = copy.deepcopy(vae)
        # Enable tiling
        patch_model(vae_copy.first_stage_model, settings)

        latent = samples["samples"]
        is_5d = latent.ndim == 5

        if is_5d:
            _, _, _, H_lat, W_lat = latent.shape
        else:
            _, _, H_lat, W_lat = latent.shape

        # For rectangular mode with scale < 1.0, crop latent to working rectangle
        # before VAE decoding so Conv2d wrapping operates at working rect boundary.
        # Use patch-aligned dimensions for consistency with the model wrapper.
        if crop and settings.mode == "Rectangular" and settings.scale < 1.0:
            patch_size = getattr(settings, '_patch_size', 1)
            work_W, work_H, margin_W, margin_H = _compute_working_size(
                W_lat, H_lat, settings, patch_size=patch_size,
            )
            if is_5d:
                latent = latent[:, :, :, margin_H:margin_H + work_H, margin_W:margin_W + work_W]
            else:
                latent = latent[:, :, margin_H:margin_H + work_H, margin_W:margin_W + work_W]

        # Decode latents to image
        image = vae_copy.decode(latent)

        # WanVAE returns 5D (B, T, H, W, C), standard VAE returns 4D (B, H, W, C)
        if image.ndim == 5:
            image = image.squeeze(1)

        if crop:
            if settings.mode == "Hexagon":
                img_h, img_w = image.shape[1], image.shape[2]
                mask = create_crop_mask(img_w, img_h, settings)
                mask_2d = mask[0, :, :, 0]
                rows = torch.any(mask_2d, dim=1)
                cols = torch.any(mask_2d, dim=0)
                row_indices = torch.where(rows)[0]
                col_indices = torch.where(cols)[0]
                rmin, rmax = row_indices[0].item(), row_indices[-1].item()
                cmin, cmax = col_indices[0].item(), col_indices[-1].item()

                # Crop to square based on hex height, centered on hex center
                hex_h = rmax - rmin + 1
                center_r = (rmin + rmax) // 2
                center_c = (cmin + cmax) // 2
                half = hex_h // 2
                sq_rmin = max(0, center_r - half)
                sq_cmin = max(0, center_c - half)
                sq_rmax = min(img_h, sq_rmin + hex_h) - 1
                sq_cmax = min(img_w, sq_cmin + hex_h) - 1
                sq_rmin = sq_rmax + 1 - hex_h
                sq_cmin = sq_cmax + 1 - hex_h

                image = image[:, sq_rmin:sq_rmax + 1, sq_cmin:sq_cmax + 1, :]
                cropped_mask = mask[:, sq_rmin:sq_rmax + 1, sq_cmin:sq_cmax + 1, :]
                image = torch.cat((image, cropped_mask.to(device=image.device)), dim=3)

        return (image,)
