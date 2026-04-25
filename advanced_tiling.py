"""
Main advanced tiling implementation
"""

from typing import Optional
import functools
import time

import torch
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from .modes import modes, Settings
from .dit_tiling import patch_dit_model, _has_conv2d, _create_content_wrapper


def _hex_remap_batch(centers_x, centers_y, size, wrap_w, wrap_h, settings):
    """Vectorized hex coordinate remapping for all pixels at once."""
    import numpy as np
    from .modes.hex import get_inverse_matrix, get_matrix

    flat_x = np.asarray(centers_x).ravel().astype(np.float64)
    flat_y = np.asarray(centers_y).ravel().astype(np.float64)

    inv_mat = get_inverse_matrix(settings)
    mat = get_matrix(settings)

    # pixel_to_hex (batch): inverse matrix multiply + divide by size
    pts = np.stack([flat_x, flat_y], axis=0)
    qr = (inv_mat @ pts) / size
    q, r = qr[0], qr[1]
    s = -q - r

    # cube_round (vectorized)
    rq, rr, rs = np.rint(q), np.rint(r), np.rint(s)
    q_diff, r_diff, s_diff = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    mask_q = (q_diff > r_diff) & (q_diff > s_diff)
    mask_r = ~mask_q & (r_diff > s_diff)
    rq = np.where(mask_q, -rr - rs, rq)
    rr = np.where(mask_r, -rq - rs, rr)

    # Fractional parts -> hex_to_pixel
    pixel = size * (mat @ np.stack([q - rq, r - rr], axis=0))
    new_x = np.rint(pixel[0]).astype(np.int64)
    new_y = np.rint(pixel[1]).astype(np.int64)

    new_x = (new_x + wrap_w // 2) % wrap_w
    new_y = (new_y + wrap_h // 2) % wrap_h
    return new_x, new_y


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

    pw, ph = padded_size
    ow, oh = original_size
    n_pixels = pw * ph

    t0 = time.perf_counter()

    if settings.mode == "Rectangular":
        pad_x = (pw - ow) // 2
        pad_y = (ph - oh) // 2
        xs = torch.arange(pw)
        ys = torch.arange(ph)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
        src_x = grid_x.flatten()
        src_y = grid_y.flatten()
        new_x = (src_x - pad_x) % ow + pad_x
        new_y = (src_y - pad_y) % oh + pad_y
        result = (tuple(src_x.tolist()), tuple(src_y.tolist()),
                  tuple(new_x.tolist()), tuple(new_y.tolist()))

    elif settings.mode == "Hexagon":
        import numpy as np
        size = max(1, round(min(ow, oh) // 2 * settings.scale))
        cx = np.arange(pw, dtype=np.float64) - pw // 2
        cy = np.arange(ph, dtype=np.float64) - ph // 2
        grid_cx, grid_cy = np.meshgrid(cx, cy, indexing='xy')
        new_x, new_y = _hex_remap_batch(grid_cx, grid_cy, size, pw, ph, settings)

        src_x = np.tile(np.arange(pw, dtype=np.int64), ph)
        src_y = np.repeat(np.arange(ph, dtype=np.int64), pw)
        result = (tuple(src_x.tolist()), tuple(src_y.tolist()),
                  tuple(new_x.tolist()), tuple(new_y.tolist()))

    else:
        # None mode: identity
        xs = torch.arange(pw)
        ys = torch.arange(ph)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
        flat_x = tuple(grid_x.flatten().tolist())
        flat_y = tuple(grid_y.flatten().tolist())
        result = (flat_x, flat_y, flat_x, flat_y)

    elapsed = time.perf_counter() - t0
    print(f"[Tiling] calculate_mapping {padded_size} ({n_pixels} px, {settings.mode}): {elapsed:.3f}s")
    return result


@functools.cache
def create_crop_mask(width: int, height: int, settings: Settings):
    """
    Crop image based on tiling settings

    :param image: Image to crop
    :param settings: Tiling settings
    :return: Cropped image
    """

    t0 = time.perf_counter()

    if settings.mode == "Hexagon":
        import numpy as np
        size = max(1, round(min(width, height) // 2 * settings.scale))
        cx = np.arange(width, dtype=np.float64) - width // 2
        cy = np.arange(height, dtype=np.float64) - height // 2
        grid_cx, grid_cy = np.meshgrid(cx, cy, indexing='xy')
        new_x, new_y = _hex_remap_batch(grid_cx, grid_cy, size, width, height, settings)

        src_x = np.tile(np.arange(width, dtype=np.int64), height)
        src_y = np.repeat(np.arange(height, dtype=np.int64), width)
        is_identity = torch.from_numpy(((new_x == src_x) & (new_y == src_y)).reshape(height, width))
        mask = torch.zeros((1, height, width, 1), dtype=torch.float32)
        mask[0, :, :, 0] = is_identity.float()
    else:
        # Rectangular/None: all pixels are in the mask
        mask = torch.ones((1, height, width, 1), dtype=torch.float32)

    elapsed = time.perf_counter() - t0
    print(f"[Tiling] create_crop_mask {width}x{height} ({settings.mode}): {elapsed:.3f}s")
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
                        "tooltip": "Working area scale relative to latent size. 0 = auto: Conv2d/Hexagon → 1.0, DiT Rectangular → 0.875. Lower values create larger margins for latent wrapping.",
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
            scale = -1.0

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

        import math

        model_copy = model.clone()

        # Resolve auto-scale: Conv2d=1.0, DiT Rectangular=0.875, DiT Hexagon=1.0
        if settings.scale < 0.0:
            is_conv2d = _has_conv2d(model_copy.model.diffusion_model)
            if is_conv2d or settings.mode == "Hexagon":
                settings.scale = 1.0
            else:
                settings.scale = round(math.sqrt(3) / 2, 3)  # 0.875

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

        t_total = time.perf_counter()

        # Patch VAE in-place instead of deepcopy (avoids copying ~167MB of weights).
        # Save original state so we can restore after decode.
        t0 = time.perf_counter()
        conv_layers = [
            layer for layer in vae.first_stage_model.modules()
            if isinstance(layer, Conv2d)
        ]
        saved = [
            (layer, layer._conv_forward, getattr(layer, 'tiling_settings', None))
            for layer in conv_layers
        ]
        patch_model(vae.first_stage_model, settings)
        print(f"[Tiling VAE] Patch {len(conv_layers)} Conv2d layers: {time.perf_counter() - t0:.4f}s")

        try:
            result = self._decode_and_crop(settings, samples, vae, crop)
        finally:
            t0 = time.perf_counter()
            for layer, orig_forward, _ in saved:
                layer._conv_forward = orig_forward
                if hasattr(layer, 'tiling_settings'):
                    del layer.tiling_settings
            print(f"[Tiling VAE] Restore: {time.perf_counter() - t0:.4f}s")

        print(f"[Tiling VAE] Total: {time.perf_counter() - t_total:.3f}s")
        return result

    def _decode_and_crop(self, settings, samples, vae, crop):
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
            from .dit_tiling import _compute_working_size
            patch_size = getattr(settings, '_patch_size', 1)
            work_W, work_H, margin_W, margin_H = _compute_working_size(
                W_lat, H_lat, settings, patch_size=patch_size,
            )
            print(f"[Tiling VAE] Latent crop: {W_lat}x{H_lat} -> {work_W}x{work_H} (margin {margin_W},{margin_H})")
            if is_5d:
                latent = latent[:, :, :, margin_H:margin_H + work_H, margin_W:margin_W + work_W]
            else:
                latent = latent[:, :, margin_H:margin_H + work_H, margin_W:margin_W + work_W]

        # Decode latents to image
        t0 = time.perf_counter()
        image = vae.decode(latent)
        print(f"[Tiling VAE] VAE decode ({W_lat}x{H_lat} latent): {time.perf_counter() - t0:.3f}s")

        # WanVAE returns 5D (B, T, H, W, C), standard VAE returns 4D (B, H, W, C)
        if image.ndim == 5:
            image = image.squeeze(1)

        if crop:
            if settings.mode == "Hexagon":
                t0 = time.perf_counter()
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
                print(f"[Tiling VAE] Hex crop ({img_w}x{img_h} -> {hex_h}x{hex_h}): {time.perf_counter() - t0:.3f}s")

        return (image,)
