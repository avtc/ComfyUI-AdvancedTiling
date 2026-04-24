"""
DiT model tiling via toroidal attention and latent wrapping.

For Hexagon mode:
- Toroidal attention: injects wrapped-neighbor K/V entries into every attention
  layer, making boundary patches structurally "see" opposite-edge content as
  spatially adjacent.
- Latent wrapping: copies source content to waste positions in the input latent
  on each denoising step (for KSampler preview).

For Rectangular mode:
- Toroidal attention: injects wrapped K/V entries from opposite edges (right<->left,
  top<->bottom).
"""

import torch
import torch.nn as nn
from torch.nn import Conv2d

from .modes import Settings


def _has_conv2d(model: nn.Module) -> bool:
    """Check if the model has any Conv2d layers (UNet vs DiT)."""
    return any(isinstance(m, Conv2d) for m in model.modules())


def _create_content_wrapper(settings: Settings):
    """
    Create a model function wrapper that copies source content to waste
    positions in the latent on each denoising step. This makes the KSampler
    preview show wrapped content instead of noise.

    Only used for Hexagon mode.

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

        cache_key = (W, H, hash(settings))
        if cache_key not in _mapping_cache:
            _mapping_cache[cache_key] = calculate_mapping(
                (W, H), (W, H), settings
            )
        mapping = _mapping_cache[cache_key]

        # Content replacement in latent
        if is_5d:
            x[:, :, :, mapping[1], mapping[0]] = x[:, :, :, mapping[3], mapping[2]]
        else:
            x[:, :, mapping[1], mapping[0]] = x[:, :, mapping[3], mapping[2]]

        return apply_model(args["input"], args["timestep"], **args["c"])

    return wrapper


def _create_lumina_wrapper(settings: Settings = None):
    """Model wrapper for Lumina: stores image shape in transformer_options
    and optionally does latent content wrapping for Hexagon mode."""
    from .advanced_tiling import calculate_mapping

    do_wrapping = settings is not None and settings.mode == "Hexagon"
    _mapping_cache = {}

    def wrapper(apply_model, args):
        x = args["input"]
        is_5d = x.ndim == 5

        if is_5d:
            _, _, _, H, W = x.shape
        else:
            _, _, H, W = x.shape

        c = dict(args["c"])
        to = c.get("transformer_options", {})
        to["tiling_img_shape"] = (H, W)
        c["transformer_options"] = to

        if do_wrapping:
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

        return apply_model(args["input"], args["timestep"], **c)

    return wrapper


def _is_lumina(diff_model) -> bool:
    """Check if the model uses Lumina/NextDiT architecture (rope_embedder instead of pe_embedder)."""
    return hasattr(diff_model, 'rope_embedder') and not hasattr(diff_model, 'pe_embedder')


def _patch_lumina_attention(model_patcher, diff_model, settings=None):
    """Apply attention-level K/V injection for Lumina/NextDiT models.

    Wraps each JointAttention.forward() in the main layers to inject boundary
    K/V with synthetic RoPE position encodings.  The wrapper is a no-op unless
    ``transformer_options["tiling_img_shape"]`` is set (by the model wrapper).
    """
    from .toroidal_attention import LuminaAttentionWrapper

    rope_embedder = diff_model.rope_embedder
    patch_size = diff_model.patch_size
    pad_tokens_multiple = getattr(diff_model, 'pad_tokens_multiple', None)

    for layer in diff_model.layers:
        LuminaAttentionWrapper(
            layer.attention, rope_embedder, patch_size, pad_tokens_multiple, settings,
        )

    wrapper = _create_lumina_wrapper(settings)
    model_patcher.set_model_unet_function_wrapper(wrapper)


def patch_dit_model(model_patcher, settings: Settings):
    """
    Apply tiling to a DiT model.

    Flux-style models (pe_embedder): attn1_patch for K/V injection.
    Lumina/NextDiT models (rope_embedder): JointAttention forward wrapper
    for K/V injection with synthetic RoPE via shift composition.

    For Hexagon mode: applies toroidal attention + latent content wrapping.
    For Rectangular mode: applies toroidal attention only.

    :param model_patcher: ComfyUI ModelPatcher instance
    :param settings: Tiling settings
    """
    diff_model = model_patcher.model.diffusion_model

    if settings.mode == "Hexagon":
        if _is_lumina(diff_model):
            _patch_lumina_attention(model_patcher, diff_model, settings)

            # Waste token replacement for hex mode
            from .toroidal_attention import LuminaWastePatch
            waste_patch = LuminaWastePatch(diff_model.patch_size, settings)
            model_patcher.set_model_patch(waste_patch, "double_block")

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
            _patch_lumina_attention(model_patcher, diff_model)

        elif hasattr(diff_model, 'pe_embedder'):
            from .toroidal_attention import RectToroidalAttentionPatch

            patch = RectToroidalAttentionPatch(diff_model.pe_embedder)
            model_patcher.set_model_attn1_patch(patch)

        else:
            raise ValueError(
                "Model does not have pe_embedder or rope_embedder. "
                "Toroidal attention requires a model with RoPE position embeddings."
            )
