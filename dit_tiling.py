"""
DiT model tiling via toroidal attention and latent wrapping.

Applies toroidal attention (K/V injection at boundary patches) combined with
latent wrapping (filling margin/waste positions with content from opposite edges
on each denoising step). The combination provides seamless infinite tiling:
- Toroidal attention makes boundary patches attend to opposite-edge content
- Latent wrapping provides spatial context at the edges for the model

For Hexagon mode:
- Toroidal attention + latent wrapping using hex coordinate mapping
- Waste positions (outside hex) filled with hex source content

For Rectangular mode:
- Toroidal attention + latent wrapping using scale-based margins
- The working rectangle is centered in the latent with margins on all sides
- Margin positions filled with content from opposite edges of working rectangle
- VAE decoder crops output to the working rectangle
"""

import math
import torch
import torch.nn as nn
from torch.nn import Conv2d

from .modes import Settings


def _has_conv2d(model: nn.Module) -> bool:
    """Check if the model has any Conv2d layers (UNet vs DiT)."""
    return any(isinstance(m, Conv2d) for m in model.modules())


def _create_content_wrapper(settings: Settings):
    """
    Create a model function wrapper that fills margin/waste positions with
    content from opposite edges on each denoising step.

    Combined with toroidal attention, this provides seamless infinite tiling
    by giving the model spatial context at the edges.

    For Hexagon mode: fills waste positions (outside hex) with hex source content.
    For Rectangular mode: fills margins with content from opposite edges of the
    working rectangle (centered in the latent).

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

        if settings.mode == "Hexagon":
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
        else:
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

        return apply_model(args["input"], args["timestep"], **args["c"])

    return wrapper


def _create_lumina_wrapper(settings: Settings = None, patch_size: int = 1):
    """Model wrapper for Lumina: applies latent content wrapping on each
    denoising step.

    The wrapping provides seamless infinite tiling by filling margin/waste
    positions with content from opposite edges, giving the model spatial
    context at the boundaries.

    :param settings: Tiling settings (None disables wrapping)
    :param patch_size: Model patch size for aligning working area to patch boundaries
    """
    from .advanced_tiling import calculate_mapping

    do_wrapping = settings is not None
    _mapping_cache = {}

    def wrapper(apply_model, args):
        if do_wrapping:
            x = args["input"]
            is_5d = x.ndim == 5

            if is_5d:
                _, _, _, H, W = x.shape
            else:
                _, _, H, W = x.shape

            if settings.mode == "Hexagon":
                cache_key = (W, H, hash(settings))
                if cache_key not in _mapping_cache:
                    _mapping_cache[cache_key] = calculate_mapping(
                        (W, H), (W, H), settings
                    )
                mapping = _mapping_cache[cache_key]
            else:
                cache_key = (W, H, hash(settings))
                if cache_key not in _mapping_cache:
                    _mapping_cache[cache_key] = calculate_mapping(
                        (W, H), (W, H), settings
                    )
                mapping = _mapping_cache[cache_key]

            if mapping is not None:
                if is_5d:
                    x[:, :, :, mapping[1], mapping[0]] = x[:, :, :, mapping[3], mapping[2]]
                else:
                    x[:, :, mapping[1], mapping[0]] = x[:, :, mapping[3], mapping[2]]


        return apply_model(args["input"], args["timestep"], **args["c"])

    return wrapper


def _is_lumina(diff_model) -> bool:
    """Check if the model uses Lumina/NextDiT architecture (rope_embedder instead of pe_embedder)."""
    return hasattr(diff_model, 'rope_embedder') and not hasattr(diff_model, 'pe_embedder')


def _patch_lumina(model_patcher, diff_model, settings=None):
    """Set up tiling for Lumina/NextDiT models.

    Two mechanisms work together:
    1. Latent wrapping (model function wrapper): fills margin/waste positions
       with content from opposite edges on each denoising step.
    2. Waste token reset (double_block_patch): resets waste tokens to their
       source content after each transformer block, preventing garbage
       accumulation from polluting attention for working-area patches.
    """
    patch_size = diff_model.patch_size

    # 1. Latent wrapping
    wrapper = _create_lumina_wrapper(settings, patch_size=patch_size)
    model_patcher.set_model_unet_function_wrapper(wrapper)

    if settings is not None:
        # 2. Waste token reset
        from .toroidal_attention import LuminaWastePatch
        waste_patch = LuminaWastePatch(patch_size, settings)
        model_patcher.set_model_double_block_patch(waste_patch)


def patch_dit_model(model_patcher, settings: Settings):
    """
    Apply tiling to a DiT model.

    Flux-style models (pe_embedder): attn1_patch for K/V injection + latent wrapping.
    Lumina/NextDiT models (rope_embedder): latent wrapping only (K/V injection
    causes boundary noise with multiplicative RoPE).

    For Hexagon mode: uses hex coordinate mapping for waste positions.
    For Rectangular mode: uses scale-based margins centered in the latent.

    :param model_patcher: ComfyUI ModelPatcher instance
    :param settings: Tiling settings
    """
    diff_model = model_patcher.model.diffusion_model

    if settings.mode == "Hexagon":
        if _is_lumina(diff_model):
            _patch_lumina(model_patcher, diff_model, settings)

        elif hasattr(diff_model, 'pe_embedder'):
            from .toroidal_attention import HexToroidalAttentionPatch

            patch = HexToroidalAttentionPatch(settings, diff_model.pe_embedder,
                                               patch_size=getattr(diff_model, 'patch_size', 1))
            model_patcher.set_model_attn1_patch(patch)

            wrapper = _create_content_wrapper(settings)
            model_patcher.set_model_unet_function_wrapper(wrapper)

        else:
            raise ValueError(
                "Model does not have pe_embedder or rope_embedder. "
                "Toroidal attention requires a model with RoPE position embeddings."
            )

    elif settings.mode == "Rectangular":
        if _is_lumina(diff_model):
            _patch_lumina(model_patcher, diff_model, settings)

        elif hasattr(diff_model, 'pe_embedder'):
            from .toroidal_attention import RectToroidalAttentionPatch

            patch = RectToroidalAttentionPatch(
                settings, diff_model.pe_embedder,
                patch_size=getattr(diff_model, 'patch_size', 1),
            )
            model_patcher.set_model_attn1_patch(patch)

            wrapper = _create_content_wrapper(settings)
            model_patcher.set_model_unet_function_wrapper(wrapper)

        else:
            raise ValueError(
                "Model does not have pe_embedder or rope_embedder. "
                "Toroidal attention requires a model with RoPE position embeddings."
            )
