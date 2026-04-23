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
    Create a model function wrapper that replaces waste content in the latent
    on each denoising step. This makes the KSampler preview show wrapped
    content instead of noise in waste areas.

    :param settings: Tiling settings
    """
    from .advanced_tiling import calculate_mapping

    _mapping_cache = {}

    def wrapper(apply_model, args):
        x = args["input"]
        is_5d = x.ndim == 5

        if is_5d:
            _, _, _, H, W = x.shape
        else:
            _, _, H, W = x.shape

        # Content replacement in latent
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

        # Model prediction
        output = apply_model(args["input"], args["timestep"], **args["c"])

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

    Optionally applies toroidal attention, latent wrapping, and position ID
    fix based on settings.

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

    # Latent content wrapping (optional, for KSampler preview)
    if settings.latent_wrapping:
        wrapper = create_latent_tiling_wrapper(settings)
        model_patcher.set_model_unet_function_wrapper(wrapper)

    # Position ID fix (optional, waste patches get source position IDs)
    if settings.position_fix:
        img_ids_patch = create_img_ids_patch(settings)
        model_patcher.set_model_post_input_patch(img_ids_patch)
