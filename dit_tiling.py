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


def _classify_hex_edge(
    x: int, y: int, width: int, height: int, settings: Settings,
) -> str | None:
    """Classify which hex edge a pixel belongs to, or None if inside."""
    from .modes.hex import pixel_to_hex, axial_round

    size = min(width, height) // 2
    q, r = pixel_to_hex(
        (x - width // 2, y - height // 2), size, settings,
    )
    rounded = axial_round((q, r))
    dq, dr = rounded
    edge_map = {
        (1, 0): "right", (-1, 0): "left",
        (0, 1): "bottom_right", (0, -1): "top_left",
        (1, -1): "top_right", (-1, 1): "bottom_left",
    }
    return edge_map.get((dq, dr))


def _create_directional_feather_mask(
    width: int, height: int, settings: Settings, blend_width: int,
) -> torch.Tensor:
    """
    Create a feathered mask that only blends inner edges on the Right,
    Top-Right, and Bottom-Right sides of the hexagon.

    Returns a [1, 1, H, W] tensor:
    - 1.0 outside hex and on untouched edges (keep model prediction)
    - 1.0 far inside the hex away from R/TR/BR boundaries (keep prediction)
    - Smooth transition from 1.0 to 0.0 approaching R/TR/BR inner boundary
    - 0.0 at the R/TR/BR boundary itself (use remapped prediction)

    :param width: Latent width
    :param height: Latent height
    :param settings: Tiling settings
    :param blend_width: Width of the transition zone in latent pixels
    """
    from .advanced_tiling import create_crop_mask

    # Binary hex mask: 1 inside, 0 outside
    hex_mask = create_crop_mask(width, height, settings)  # [1, H, W, 1]
    hex_mask = hex_mask.permute(0, 3, 1, 2).squeeze(0).squeeze(0)  # [H, W]

    # Identify R/TR/BR waste pixels
    r_tr_br_mask = torch.zeros(height, width, dtype=torch.float32)
    for y in range(height):
        for x in range(width):
            edge = _classify_hex_edge(x, y, width, height, settings)
            if edge in ("right", "top_right", "bottom_right"):
                r_tr_br_mask[y, x] = 1.0

    # Approximate distance from each pixel to nearest R/TR/BR waste pixel
    # using iterative morphological erosion (no scipy dependency).
    # For each pixel, count how many dilations of the waste region
    # are needed to reach it.
    waste = r_tr_br_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    dist = torch.zeros(1, 1, height, width, dtype=torch.float32)
    current = waste.clone()
    for step in range(1, blend_width + 1):
        dilated = F.max_pool2d(current, 3, stride=1, padding=1)
        newly_reached = (dilated > 0) & (current == 0)
        dist[newly_reached] = step
        current = dilated.clamp(0, 1)

    # Build the feather mask:
    # - Outside hex: 1.0 (no blend)
    # - Inside hex, far from R/TR/BR boundary: 1.0 (no blend)
    # - Inside hex, near R/TR/BR boundary: transition from 1.0 to 0.0
    mask = torch.ones(1, 1, height, width, dtype=torch.float32)
    inside = hex_mask.bool()
    reached = dist[0, 0] > 0
    transition = reached & (dist[0, 0] < blend_width) & inside
    mask[0, 0][transition] = dist[0, 0][transition] / blend_width
    # Pixels right at the R/TR/BR boundary: inside, adjacent to R/TR/BR waste
    # (reached by first dilation step but inside hex)
    first_ring = (dist[0, 0] == 1) & inside
    # Also catch inside pixels directly adjacent to waste (dist would be 1)
    adjacent_to_waste = F.max_pool2d(waste, 3, stride=1, padding=1)
    at_boundary = (adjacent_to_waste[0, 0] > 0) & inside & (waste[0, 0] == 0)
    mask[0, 0][at_boundary] = 0.0

    return mask.clamp(0, 1)


def create_latent_tiling_wrapper(settings: Settings, blend_width: int = 64):
    """
    Create a model function wrapper that:
    1. Replaces waste content in the latent (for preview)
    2. Runs the model
    3. Applies noise-level enforcement on the output using a directional
       feather mask (only on R/TR/BR inner edges)

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

        # Step 3: Noise-level enforcement — blend R/TR/BR inner edges
        if blend_width > 0:
            mask_key = (W, H, hash(settings), blend_width)
            if mask_key not in _mask_cache:
                _mask_cache[mask_key] = _create_directional_feather_mask(
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

            # Blend: feather=1 keeps model prediction, feather=0 uses remapped
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
