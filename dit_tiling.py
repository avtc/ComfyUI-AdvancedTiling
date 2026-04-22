"""
DiT model tiling via latent content replacement and per-block hidden state
replacement.

Works with any DiT model that supports the model_function_wrapper,
post_input, and double_block_patch hooks, including Qwen Image.

Three mechanisms work together:
1. Latent content replacement: copies source content to waste positions in
   the input latent on each denoising step (model wrapper hook).
2. Position ID fix: replaces waste patches' position IDs with their source
   positions so RoPE encodes them at the opposite edge (post_input hook).
3. Per-block hidden state replacement: after each transformer block, replaces
   waste patches' hidden states with source patches' hidden states
   (double_block hook). This forces the model to process source content at
   waste positions at every layer, analogous to how Conv2d wrapping works.
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


def create_latent_tiling_wrapper(settings: Settings, shared: dict):
    """
    Create a model function wrapper that copies source content to waste
    positions in the input latent on each denoising step.

    Also stores latent dimensions in shared dict for the double_block_patch.

    :param settings: Tiling settings
    :param shared: Shared state dict between hooks
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

        shared["H"] = H
        shared["W"] = W

        cache_key = (W, H, hash(settings))
        if cache_key not in _cache:
            _cache[cache_key] = calculate_mapping(
                (W, H), (W, H), settings
            )

        mapping = _cache[cache_key]

        if is_5d:
            x[:, :, :, mapping[1], mapping[0]] = x[:, :, :, mapping[3], mapping[2]]
        else:
            x[:, :, mapping[1], mapping[0]] = x[:, :, mapping[3], mapping[2]]

        return apply_model(args["input"], args["timestep"], **args["c"])

    return wrapper


def create_double_block_patch(settings: Settings, shared: dict):
    """
    Create a double_block_patch that replaces waste patches' hidden states
    with source patches' hidden states after each transformer block.

    This is the DiT equivalent of Conv2d per-layer wrapping: at every layer,
    the waste region is forced to contain source content, so the model
    processes wrapped information throughout the network.

    :param settings: Tiling settings
    :param shared: Shared state dict (expects 'H', 'W' from wrapper)
    :return: Patch function for set_model_double_block_patch
    """
    _cache = {}

    def block_patch(img, txt, extra_options):
        H = shared.get("H")
        W = shared.get("W")
        if H is None or W is None:
            return img, txt

        num_patches = img.shape[1]
        patch_size = int(round((H * W / num_patches) ** 0.5))
        h_patches = H // patch_size
        w_patches = W // patch_size

        cache_key = (h_patches, w_patches, hash(settings))
        if cache_key not in _cache:
            _cache[cache_key] = _build_waste_mapping(
                h_patches, w_patches, settings,
            )

        waste_idx, source_idx = _cache[cache_key]
        img[:, waste_idx] = img[:, source_idx]

        return img, txt

    return block_patch


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

    Three hooks work together:
    1. Model wrapper: content replacement in latent
    2. Post_input: position ID fix for waste patches
    3. Double_block: hidden state replacement for waste patches at every layer

    :param model_patcher: ComfyUI ModelPatcher instance
    :param settings: Tiling settings
    """
    shared = {}

    wrapper = create_latent_tiling_wrapper(settings, shared)
    model_patcher.set_model_unet_function_wrapper(wrapper)

    img_ids_patch = create_img_ids_patch(settings)
    model_patcher.set_model_post_input_patch(img_ids_patch)

    block_patch = create_double_block_patch(settings, shared)
    model_patcher.set_model_double_block_patch(block_patch)
