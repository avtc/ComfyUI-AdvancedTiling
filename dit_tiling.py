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


def _compute_working_size(W, H, settings, patch_size=1):
    """Compute working area dimensions from scale + min_margin, centered in the latent.

    When patch_size > 1, computes at the patch level and converts to pixel
    coordinates, ensuring alignment with patch boundaries. This prevents
    misalignment between pixel-level latent wrapping and patch-level operations
    (waste token reset, attention).

    :param W: Latent width in pixels
    :param H: Latent height in pixels
    :param settings: Tiling settings with scale, min_margin, and divisible_by
    :param patch_size: Model patch size (1 = no alignment, for UNet/Conv2d)
    :return: (work_W, work_H, margin_W, margin_H) in pixel coordinates
    """
    scale = settings.scale
    min_margin = getattr(settings, 'min_margin', 0)
    divisible_by = getattr(settings, 'divisible_by', 16)

    if scale >= 1.0 and min_margin == 0:
        return W, H, 0, 0

    # Convert image-pixel alignment to latent-pixel alignment (assume 8x VAE)
    latent_align = max(1, divisible_by // 8)

    if patch_size > 1:
        h_patches = H // patch_size
        w_patches = W // patch_size

        if scale < 1.0:
            scale_work_h = max(1, round(h_patches * scale))
            scale_work_w = max(1, round(w_patches * scale))
            scale_margin_h = (h_patches - scale_work_h) // 2
            scale_margin_w = (w_patches - scale_work_w) // 2
        else:
            scale_margin_h = 0
            scale_margin_w = 0

        margin_h = max(scale_margin_h, min_margin)
        margin_w = max(scale_margin_w, min_margin)
        work_h = h_patches - 2 * margin_h
        work_w = w_patches - 2 * margin_w

        # Round down to alignment
        patch_align = max(latent_align // patch_size, 1)
        work_h = max(patch_align, (work_h // patch_align) * patch_align)
        work_w = max(patch_align, (work_w // patch_align) * patch_align)

        # Recompute margin from rounded work
        margin_h = (h_patches - work_h) // 2
        margin_w = (w_patches - work_w) // 2

        if work_h < 1 or work_w < 1:
            return W, H, 0, 0

        return (work_w * patch_size, work_h * patch_size,
                margin_w * patch_size, margin_h * patch_size)

    if scale < 1.0:
        scale_work_W = max(1, round(W * scale))
        scale_work_H = max(1, round(H * scale))
        scale_margin_W = (W - scale_work_W) // 2
        scale_margin_H = (H - scale_work_H) // 2
    else:
        scale_margin_W = 0
        scale_margin_H = 0

    margin_W = max(scale_margin_W, min_margin)
    margin_H = max(scale_margin_H, min_margin)
    work_W = W - 2 * margin_W
    work_H = H - 2 * margin_H

    # Round down to alignment
    work_W = max(latent_align, (work_W // latent_align) * latent_align)
    work_H = max(latent_align, (work_H // latent_align) * latent_align)

    # Recompute margin from rounded work
    margin_W = (W - work_W) // 2
    margin_H = (H - work_H) // 2

    if work_W < 1 or work_H < 1:
        return W, H, 0, 0

    return work_W, work_H, margin_W, margin_H


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
            work_W, work_H, margin_W, margin_H = _compute_working_size(W, H, settings)
            if margin_W > 0 or margin_H > 0:
                cache_key = (W, H, settings.scale, getattr(settings, 'min_margin', 0), getattr(settings, 'divisible_by', 16))
                if cache_key not in _mapping_cache:
                    _mapping_cache[cache_key] = calculate_mapping(
                        (work_W, work_H), (W, H), settings
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
                work_W, work_H, margin_W, margin_H = _compute_working_size(
                    W, H, settings, patch_size=patch_size,
                )
                if margin_W > 0 or margin_H > 0:
                    cache_key = (W, H, work_W, work_H)
                    if cache_key not in _mapping_cache:
                        _mapping_cache[cache_key] = calculate_mapping(
                            (work_W, work_H), (W, H), settings
                        )
                    mapping = _mapping_cache[cache_key]
                else:
                    mapping = None

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
    settings._patch_size = patch_size

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

            patch = HexToroidalAttentionPatch(settings, diff_model.pe_embedder)
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
                diff_model.pe_embedder, scale=settings.scale,
                min_margin=getattr(settings, 'min_margin', 0),
                divisible_by=getattr(settings, 'divisible_by', 16),
            )
            model_patcher.set_model_attn1_patch(patch)

            wrapper = _create_content_wrapper(settings)
            model_patcher.set_model_unet_function_wrapper(wrapper)

        else:
            raise ValueError(
                "Model does not have pe_embedder or rope_embedder. "
                "Toroidal attention requires a model with RoPE position embeddings."
            )
