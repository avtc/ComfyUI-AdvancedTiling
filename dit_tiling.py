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
    _wrapper_call_count = 0

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

            settings._current_img_shape = (H, W)

            nonlocal _wrapper_call_count
            _wrapper_call_count += 1
            if _wrapper_call_count <= 3:
                h_patches = H // patch_size if patch_size > 1 else H
                w_patches = W // patch_size if patch_size > 1 else W
                print(f"[TILING-DEBUG] wrapper #{_wrapper_call_count}: "
                      f"mode={settings.mode}, latent=({W}x{H}), "
                      f"patches=({w_patches}x{h_patches}), patch_size={patch_size}, "
                      f"scale={settings.scale}, mapping={'yes' if mapping is not None else 'no'}")

        return apply_model(args["input"], args["timestep"], **args["c"])

    return wrapper


def _is_lumina(diff_model) -> bool:
    """Check if the model uses Lumina/NextDiT architecture (rope_embedder instead of pe_embedder)."""
    return hasattr(diff_model, 'rope_embedder') and not hasattr(diff_model, 'pe_embedder')


def _patch_lumina(model_patcher, diff_model, settings=None):
    """Set up tiling for Lumina/NextDiT models.

    Uses latent wrapping (model function wrapper) plus a double_block waste
    token reset patch. When lumina_kv_injection is enabled, also wraps each
    JointAttention in diff_model.layers to inject boundary K/V at virtual
    adjacent positions with correct synthetic freqs_cis.

    The waste patch resets margin/waste tokens to their source content after
    each transformer block, preventing garbage accumulation from polluting
    attention for working-area patches.
    """
    print(f"[TILING-DEBUG] _patch_lumina called: mode={settings.mode if settings else None}, "
          f"scale={settings.scale if settings else None}, "
          f"patch_size={diff_model.patch_size}, "
          f"kv_injection={settings.lumina_kv_injection if settings else None}")

    patch_size = diff_model.patch_size
    settings._patch_size = patch_size

    wrapper = _create_lumina_wrapper(settings, patch_size=patch_size)
    model_patcher.set_model_unet_function_wrapper(wrapper)

    if settings is not None:
        from .toroidal_attention import LuminaWastePatch
        waste_patch = LuminaWastePatch(diff_model.patch_size, settings)
        model_patcher.set_model_double_block_patch(waste_patch)

        if settings.lumina_kv_injection:
            from .toroidal_attention import LuminaKVInjectionWrapper
            rope_embedder = diff_model.rope_embedder
            patch_size = diff_model.patch_size
            pad_tokens_multiple = getattr(diff_model, 'pad_tokens_multiple', None)

            n_wrapped = 0
            for layer in diff_model.layers:
                attn = layer.attention
                LuminaKVInjectionWrapper(
                    attn, rope_embedder, patch_size, pad_tokens_multiple, settings
                )
                n_wrapped += 1
            print(f"[TILING-DEBUG] K/V injection ENABLED: wrapped {n_wrapped} attention layers, "
                  f"patch_size={patch_size}, pad_tokens_multiple={pad_tokens_multiple}")
        else:
            print(f"[TILING-DEBUG] K/V injection DISABLED")


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
