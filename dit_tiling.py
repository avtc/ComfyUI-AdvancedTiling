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
    """Compute working area dimensions from scale, centered in the latent.

    When patch_size > 1, computes at the patch level and converts to pixel
    coordinates, ensuring alignment with patch boundaries. This prevents
    misalignment between pixel-level latent wrapping and patch-level operations
    (waste token reset, attention).

    :param W: Latent width in pixels
    :param H: Latent height in pixels
    :param settings: Tiling settings with scale
    :param patch_size: Model patch size (1 = no alignment, for UNet/Conv2d)
    :return: (work_W, work_H, margin_W, margin_H) in pixel coordinates
    """
    scale = settings.scale
    if scale >= 1.0:
        return W, H, 0, 0

    if patch_size > 1:
        h_patches = H // patch_size
        w_patches = W // patch_size
        work_h = max(1, round(h_patches * scale))
        work_w = max(1, round(w_patches * scale))
        margin_h = (h_patches - work_h) // 2
        margin_w = (w_patches - work_w) // 2
        return (work_w * patch_size, work_h * patch_size,
                margin_w * patch_size, margin_h * patch_size)

    work_W = max(1, round(W * scale))
    work_H = max(1, round(H * scale))
    margin_W = (W - work_W) // 2
    margin_H = (H - work_H) // 2
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
                cache_key = (W, H, settings.scale)
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


def _compute_boundary_blend_indices(W, H, margin_W, margin_H, work_W, work_H, patch_size):
    """Compute boundary blend mapping: working-area edge pixels → opposite edge pixels.

    For each pixel in a feathered zone at the edge of the working area, compute
    its corresponding pixel on the opposite edge. Returns destination and source
    index arrays plus a weight array (1.0 at edge, 0.0 at interior).

    :return: (dest_indices, src_indices, weights) or None if no blending needed
    """
    blend_patches = 2
    blend_px = blend_patches * patch_size

    if blend_px <= 0 or margin_W <= 0 or margin_H <= 0:
        return None

    dest_y, dest_x, src_y, src_x, weights = [], [], [], [], []

    for y in range(margin_H, margin_H + work_H):
        for x in range(margin_W, margin_W + work_W):
            alpha = 0.0

            dist_top = y - margin_H
            dist_bottom = (margin_H + work_H - 1) - y
            dist_left = x - margin_W
            dist_right = (margin_W + work_W - 1) - x

            if dist_top < blend_px:
                alpha = max(alpha, 1.0 - dist_top / blend_px)
                sy = margin_H + work_H - 1 - dist_top
            elif dist_bottom < blend_px:
                alpha = max(alpha, 1.0 - dist_bottom / blend_px)
                sy = margin_H + dist_bottom
            else:
                sy = y

            if dist_left < blend_px:
                alpha = max(alpha, 1.0 - dist_left / blend_px)
                sx = margin_W + work_W - 1 - dist_left
            elif dist_right < blend_px:
                alpha = max(alpha, 1.0 - dist_right / blend_px)
                sx = margin_W + dist_right
            else:
                sx = x

            if alpha > 0.0:
                dest_y.append(y)
                dest_x.append(x)
                src_y.append(sy)
                src_x.append(sx)
                weights.append(alpha)

    if not dest_y:
        return None

    return (
        torch.tensor(dest_y, dtype=torch.long),
        torch.tensor(dest_x, dtype=torch.long),
        torch.tensor(src_y, dtype=torch.long),
        torch.tensor(src_x, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
    )


def _create_lumina_wrapper(settings: Settings = None, patch_size: int = 1):
    """Model wrapper for Lumina/Z-Image: latent wrapping + boundary blend.

    The wrapping fills margin/waste positions with content from opposite edges,
    giving the model spatial context at the boundaries.

    If boundary_blend is enabled, after the model forward pass the noise
    prediction at working-area edges is blended with the opposite edge to
    smooth position-mismatch artifacts.

    :param settings: Tiling settings (None disables wrapping)
    :param patch_size: Model patch size for aligning working area to patch boundaries
    """
    from .advanced_tiling import calculate_mapping

    do_wrapping = settings is not None
    _mapping_cache = {}
    _blend_cache = {}

    def wrapper(apply_model, args):
        if do_wrapping:
            x = args["input"]
            is_5d = x.ndim == 5

            if is_5d:
                _, _, _, H, W = x.shape
            else:
                _, _, H, W = x.shape

            # Store timestep for waste patch (timestep_decay)
            settings._current_timestep = args["timestep"]

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

            # Model forward pass
            result = apply_model(args["input"], args["timestep"], **args["c"])

            # Boundary blend: smooth noise prediction at working-area edges
            if settings.boundary_blend and settings.mode == "Rectangular" and mapping is not None:
                blend_key = (W, H, work_W, work_H)
                if blend_key not in _blend_cache:
                    _blend_cache[blend_key] = _compute_boundary_blend_indices(
                        W, H, margin_W, margin_H, work_W, work_H, patch_size,
                    )
                blend = _blend_cache[blend_key]

                if blend is not None:
                    dy, dx, sy, sx, w = blend
                    w = w.to(result.device).view(1, 1, -1)
                    if is_5d:
                        flat_dst = result[:, :, :, dy, dx]
                        flat_src = result[:, :, :, sy, sx]
                        blended = flat_dst * (1 - w) + flat_src * w
                        result[:, :, :, dy, dx] = blended
                    else:
                        flat_dst = result[:, :, dy, dx]
                        flat_src = result[:, :, sy, sx]
                        blended = flat_dst * (1 - w) + flat_src * w
                        result[:, :, dy, dx] = blended

            return result

        return apply_model(args["input"], args["timestep"], **args["c"])

    return wrapper


def _is_lumina(diff_model) -> bool:
    """Check if the model uses Lumina/NextDiT architecture (rope_embedder instead of pe_embedder)."""
    return hasattr(diff_model, 'rope_embedder') and not hasattr(diff_model, 'pe_embedder')


def _patch_lumina(model_patcher, diff_model, settings=None):
    """Set up tiling for Lumina/NextDiT models.

    Mechanisms:
    1. Latent wrapping (model function wrapper): fills margin/waste positions
       with content from opposite edges on each denoising step.
    2. Waste token reset (double_block_patch): resets waste tokens to their
       source content after each transformer block, preventing garbage
       accumulation from polluting attention for working-area patches.
    3. Optional rope_fix: replaces freqs_cis at waste positions with source
       positions' freqs_cis, fixing position-content mismatch for RoPE.
    4. Optional timestep_decay: scales waste token reset by timestep-dependent
       factor (full influence early, reduced late).
    5. Optional boundary_blend: smooths noise prediction at working-area edges
       by blending with the opposite edge.
    """
    patch_size = diff_model.patch_size
    settings._patch_size = patch_size

    # 1. Latent wrapping (+ boundary_blend if enabled)
    wrapper = _create_lumina_wrapper(settings, patch_size=patch_size)
    model_patcher.set_model_unet_function_wrapper(wrapper)

    if settings is not None:
        # 2. Waste token reset (+ rope_fix, timestep_decay if enabled)
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

            patch = RectToroidalAttentionPatch(diff_model.pe_embedder, scale=settings.scale)
            model_patcher.set_model_attn1_patch(patch)

            wrapper = _create_content_wrapper(settings)
            model_patcher.set_model_unet_function_wrapper(wrapper)

        else:
            raise ValueError(
                "Model does not have pe_embedder or rope_embedder. "
                "Toroidal attention requires a model with RoPE position embeddings."
            )
