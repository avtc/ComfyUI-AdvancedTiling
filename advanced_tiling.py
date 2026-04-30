"""
Main advanced tiling implementation
"""

from typing import Optional
import functools
import math

import torch
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from .modes import MODE_NAMES, Settings, ResolvedSettings
from .dit_tiling import patch_dit_model, _has_conv2d


def _hex_remap_batch(centers_x, centers_y, size, wrap_w, wrap_h, rotation):
    """Vectorized hex coordinate remapping for all pixels at once."""
    import numpy as np
    from .modes.hex import get_inverse_matrix, get_matrix

    flat_x = np.asarray(centers_x).ravel().astype(np.float64)
    flat_y = np.asarray(centers_y).ravel().astype(np.float64)

    inv_mat = get_inverse_matrix(rotation)
    mat = get_matrix(rotation)

    # pixel_to_hex (batch): inverse matrix multiply + divide by size
    pts = np.stack([flat_x, flat_y], axis=0)
    qr = (inv_mat @ pts) / size
    q, r = qr[0], qr[1]
    s = -q - r

    # cube_round (vectorized) — floor(x+0.5) for consistent half-up rounding
    _half_up = lambda v: np.floor(v + 0.5)
    rq, rr, rs = _half_up(q), _half_up(r), _half_up(s)
    q_diff, r_diff, s_diff = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    mask_q = (q_diff > r_diff) & (q_diff > s_diff)
    mask_r = ~mask_q & (r_diff > s_diff)
    rq = np.where(mask_q, -rr - rs, rq)
    rr = np.where(mask_r, -rq - rs, rr)

    # Fractional parts -> hex_to_pixel
    pixel = size * (mat @ np.stack([q - rq, r - rr], axis=0))
    # Use floor(x+0.5) instead of rint to avoid banker's rounding on exact 0.5:
    # when hex_size is an integer at a VAE decoder resolution, size*3/2 becomes a
    # half-integer, causing pixel offsets to land exactly on X.5 boundaries where
    # np.rint rounds to even — producing discontinuities in the wrapped image.
    new_x = np.floor(pixel[0] + 0.5).astype(np.int64)
    new_y = np.floor(pixel[1] + 0.5).astype(np.int64)

    new_x = (new_x + wrap_w // 2) % wrap_w
    new_y = (new_y + wrap_h // 2) % wrap_h
    return new_x, new_y


@functools.cache
def calculate_mapping(
    original_size: tuple[int, int], padded_size: tuple[int, int],
    resolved: ResolvedSettings,
):
    """
    Calculate mapping for pixels outside of the mask

    :param original_size: Original size of the image
    :param padded_size: Padded size of the image
    :param resolved: Resolved tiling settings with pre-computed values
    :return: Mapping of pixels
    """
    pw, ph = padded_size
    ow, oh = original_size

    if resolved.mode == "Rectangular":
        # Scale working area to actual tensor resolution.
        # At diffusion latent resolution ow=img_w/vae_factor, scale=1.0.
        # VAE decoder Conv2d layers at 2x/4x/8x get proportionally scaled.
        work_w, work_h = resolved.work_at(ow, oh)
        cx, cy = pw / 2.0, ph / 2.0

        xs = torch.arange(pw, dtype=torch.float64)
        ys = torch.arange(ph, dtype=torch.float64)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')

        rel_x = grid_x - cx
        rel_y = grid_y - cy

        new_x = cx + torch.fmod(torch.fmod(rel_x + work_w / 2, work_w) + work_w, work_w) - work_w / 2
        new_y = cy + torch.fmod(torch.fmod(rel_y + work_h / 2, work_h) + work_h, work_h) - work_h / 2

        new_x = torch.floor(new_x + 0.5).to(torch.long)
        new_y = torch.floor(new_y + 0.5).to(torch.long)
        src_x = grid_x.flatten().to(torch.long)
        src_y = grid_y.flatten().to(torch.long)
        new_x = new_x.flatten()
        new_y = new_y.flatten()

        non_id = (src_x != new_x) | (src_y != new_y)
        result = (src_x[non_id], src_y[non_id], new_x[non_id], new_y[non_id])

    elif resolved.mode == "Hexagon":
        import numpy as np
        # Scale hex size to actual tensor resolution.
        size = resolved.hex_size_at(ow, oh)
        cx = np.arange(pw, dtype=np.float64) - pw // 2
        cy = np.arange(ph, dtype=np.float64) - ph // 2
        grid_cx, grid_cy = np.meshgrid(cx, cy, indexing='xy')
        new_x, new_y = _hex_remap_batch(grid_cx, grid_cy, size, pw, ph, resolved.rotation)

        src_x = np.tile(np.arange(pw, dtype=np.int64), ph)
        src_y = np.repeat(np.arange(ph, dtype=np.int64), pw)
        # Only keep pixels that actually change position
        non_id = (src_x != new_x) | (src_y != new_y)
        result = (torch.from_numpy(src_x[non_id]), torch.from_numpy(src_y[non_id]),
                  torch.from_numpy(new_x[non_id]), torch.from_numpy(new_y[non_id]))

    else:
        # None mode: no remapping needed
        empty = torch.tensor([], dtype=torch.long)
        result = (empty, empty.clone(), empty.clone(), empty.clone())

    return result


@functools.cache
def create_crop_mask(width: int, height: int, resolved: ResolvedSettings):
    """
    Create crop mask based on resolved tiling settings.

    :param width: Width of the image
    :param height: Height of the image
    :param resolved: Resolved tiling settings with pre-computed values
    :return: Crop mask tensor (1, H, W, 1)
    """
    if resolved.mode == "Hexagon":
        import numpy as np

        size = resolved.hex_size_img
        cx = np.arange(width, dtype=np.float64) - width // 2
        cy = np.arange(height, dtype=np.float64) - height // 2
        grid_cx, grid_cy = np.meshgrid(cx, cy, indexing='xy')
        new_x, new_y = _hex_remap_batch(grid_cx, grid_cy, size, width, height, resolved.rotation)

        src_x = np.tile(np.arange(width, dtype=np.int64), height)
        src_y = np.repeat(np.arange(height, dtype=np.int64), width)
        is_identity = ((new_x == src_x) & (new_y == src_y)).reshape(height, width)

        mask = torch.zeros((1, height, width, 1), dtype=torch.float32)
        mask[0, :, :, 0] = torch.from_numpy(is_identity.astype(np.float32))

    elif resolved.mode == "Rectangular":
        work_w, work_h = resolved.work_img_w, resolved.work_img_h
        cx, cy = width / 2.0, height / 2.0
        left = cx - work_w / 2
        right = cx + work_w / 2
        top = cy - work_h / 2
        bottom = cy + work_h / 2

        xs = torch.arange(width, dtype=torch.float64)
        ys = torch.arange(height, dtype=torch.float64)
        inside_x = (xs >= left) & (xs < right)
        inside_y = (ys >= top) & (ys < bottom)
        is_identity = inside_y.unsqueeze(1) & inside_x.unsqueeze(0)

        mask = torch.zeros((1, height, width, 1), dtype=torch.float32)
        mask[0, :, :, 0] = is_identity.float()

    else:
        mask = torch.ones((1, height, width, 1), dtype=torch.float32)

    return mask


def compute_crop_bounds(img_w: int, img_h: int, resolved: ResolvedSettings):
    """Compute crop bounding box (rmin, rmax, cmin, cmax) inclusive.

    :param img_w: Image width
    :param img_h: Image height
    :param resolved: Resolved tiling settings
    :return: (rmin, rmax, cmin, cmax) inclusive pixel indices
    """
    if resolved.mode == "Rectangular" and resolved.margin_img_w > 0:
        crop_w = int(math.floor(resolved.work_img_w + 0.5))
        crop_h = int(math.floor(resolved.work_img_h + 0.5))
        center_c = img_w // 2
        center_r = img_h // 2
        cmin = center_c - crop_w // 2
        cmax = cmin + crop_w - 1
        rmin = center_r - crop_h // 2
        rmax = rmin + crop_h - 1
    elif resolved.mode == "Hexagon":
        hex_side = int(math.floor(2 * resolved.hex_size_img + 0.5))
        center_r = img_h // 2
        center_c = img_w // 2
        half = hex_side // 2
        rmin = max(0, center_r - half)
        cmin = max(0, center_c - half)
        rmax = min(img_h, rmin + hex_side) - 1
        cmax = min(img_w, cmin + hex_side) - 1
        rmin = max(0, rmax + 1 - hex_side)
        cmin = max(0, cmax + 1 - hex_side)
    else:
        rmin, rmax, cmin, cmax = 0, img_h - 1, 0, img_w - 1
    return rmin, rmax, cmin, cmax


def _mask_bounding_box(mask: torch.Tensor) -> tuple[int, int, int, int]:
    """Find the bounding box of non-zero region in a crop mask.

    :param mask: Shape (1, H, W, 1)
    :return: (rmin, rmax, cmin, cmax) inclusive indices
    """
    mask_2d = mask[0, :, :, 0]
    rows = torch.any(mask_2d, dim=1)
    cols = torch.any(mask_2d, dim=0)
    row_indices = torch.where(rows)[0]
    col_indices = torch.where(cols)[0]
    return (
        row_indices[0].item(), row_indices[-1].item(),
        col_indices[0].item(), col_indices[-1].item(),
    )


def _crop_with_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    rmin: int, rmax: int, cmin: int, cmax: int,
) -> torch.Tensor:
    """Slice image and mask to bounding box, concatenate mask as alpha channel.

    :param image: Shape (B, H, W, C)
    :param mask: Shape (1, H, W, 1)
    :return: Image with mask appended as last channel
    """
    image = image[:, rmin:rmax + 1, cmin:cmax + 1, :]
    cropped_mask = mask[:, rmin:rmax + 1, cmin:cmax + 1, :]
    return torch.cat((image, cropped_mask.to(device=image.device)), dim=3)


def patch_model(model, resolved: ResolvedSettings):
    """
    Patch model to perform tiling - in place!

    :param model: Model to patch
    :param resolved: Resolved tiling settings
    """
    for layer in [layer for layer in model.modules() if isinstance(layer, Conv2d)]:
        # pylint: disable=protected-access, no-value-for-parameter
        layer._conv_forward = tiling_conv.__get__(layer, Conv2d)
        layer.tiling_resolved = resolved
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
        self.tiling_resolved,
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
                "mode": (MODE_NAMES, {
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
                        "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                        "tooltip": "Working area scale relative to latent size. 0 = auto: Conv2d/Hexagon → 1.0, DiT Rectangular → 0.875. Set to 1.0 and use min_margin for absolute pixel control.",
                    },
                ),
                "min_margin": (
                    "INT",
                    {
                        "default": -1, "min": -1, "max": 512, "step": 1,
                        "tooltip": "Minimum margin in latent pixels. -1 = auto (4 for DiT Rectangular, 0 otherwise). Rectangular: floors the margin from scale. Hexagon: reduces hex radius. Set scale=1.0 + min_margin=N for exact absolute margin.",
                    },
                ),
                "divisible_by": (
                    "INT",
                    {
                        "default": 1, "min": 1, "max": 256, "step": 1,
                        "tooltip": "Round working area down to multiples of this value. Affects wrapping and crop. 1 = no rounding.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("ADVANCED_TILING_SETTINGS",)
    RETURN_NAMES = ("SETTINGS",)
    FUNCTION = "run"

    def run(self, mode, rotation, scale, min_margin, divisible_by):
        """
        Creates tiling settings from node inputs
        """

        settings = Settings(mode, rotation, scale, min_margin, divisible_by)

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
                "vae": ("VAE",),
                "latent": ("LATENT",),
            },
        }

    CATEGORY = "conditioning"
    RETURN_TYPES = ("MODEL", "RESOLVED_TILING_SETTINGS")
    RETURN_NAMES = ("MODEL", "resolved_settings")
    FUNCTION = "run"

    def run(self, settings, model, vae, latent):
        """
        Does the actual patching of the model
        """

        model_copy = model.clone()

        diff_model = model_copy.model.diffusion_model
        is_conv2d = _has_conv2d(diff_model)
        vae_factor = vae.spacial_compression_decode()
        patch_size = 1 if is_conv2d else getattr(diff_model, 'patch_size', 1)

        latent_tensor = latent["samples"]
        _, _, H_lat, W_lat = latent_tensor.shape
        img_w = W_lat * vae_factor
        img_h = H_lat * vae_factor

        resolved = settings._resolve_auto(is_conv2d, vae_factor, patch_size, img_w, img_h)

        if resolved.mode == "None":
            return (model_copy, resolved)

        if is_conv2d:
            patch_model(model_copy.model, resolved)
        else:
            patch_dit_model(model_copy, resolved)

        return (model_copy, resolved)



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
                "resolved_settings": ("RESOLVED_TILING_SETTINGS",),
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "crop": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "latent"

    def run(self, resolved_settings, samples, vae, crop):
        """
        Decode latents to image with tiling
        Optionally crop the image based on tiling settings

        For Hexagon mode: applies alpha mask for hex-shaped cropping.
        For Rectangular mode with margin: decodes full latent
        then crops using mask-based post-decode crop.

        Settings must be pre-resolved (from AdvancedTiling node output).
        """

        if resolved_settings.mode == "None":
            image = vae.decode(samples["samples"])
            if image.ndim == 5:
                image = image.squeeze(1)
            return (image,)

        # Patch VAE in-place instead of deepcopy (avoids copying ~167MB of weights).
        # Save original state so we can restore after decode.
        conv_layers = [
            layer for layer in vae.first_stage_model.modules()
            if isinstance(layer, Conv2d)
        ]
        saved = [
            (layer, layer._conv_forward, getattr(layer, 'tiling_resolved', None))
            for layer in conv_layers
        ]

        # Use settings as-is — they were resolved by AdvancedTiling.run()
        # to match the generation model's working area.
        patch_model(vae.first_stage_model, resolved_settings)

        try:
            result = self._decode_and_crop(resolved_settings, samples, vae, crop)
        finally:
            # Restore original state — patch_model added tiling_resolved to
            # every Conv2d layer, so deleting it is always safe here.
            for layer, orig_forward, _ in saved:
                layer._conv_forward = orig_forward
                if hasattr(layer, 'tiling_resolved'):
                    del layer.tiling_resolved

        return result

    def _decode_and_crop(self, resolved_settings, samples, vae, crop):
        latent = samples["samples"]

        # Decode full latent — crop after decode using mask
        image = vae.decode(latent)

        if image.ndim == 5:
            image = image.squeeze(1)

        if crop:
            img_h, img_w = image.shape[1], image.shape[2]
            mask = create_crop_mask(img_w, img_h, resolved_settings)

            if resolved_settings.mode != "None":
                rmin, rmax, cmin, cmax = compute_crop_bounds(img_w, img_h, resolved_settings)
                image = _crop_with_mask(image, mask, rmin, rmax, cmin, cmax)

        return (image,)
