"""
DiT model tiling via toroidal attention.

Uses attn1_patch to inject wrapped-neighbor K/V entries into every attention
layer, making boundary patches structurally "see" opposite-edge content as
spatially adjacent. Works with any DiT model that supports the attn1_patch
hook, including Qwen Image.
"""

import torch.nn as nn
from torch.nn import Conv2d

from .modes import Settings


def _has_conv2d(model: nn.Module) -> bool:
    """Check if the model has any Conv2d layers (UNet vs DiT)."""
    return any(isinstance(m, Conv2d) for m in model.modules())


def patch_dit_model(model_patcher, settings: Settings):
    """
    Apply toroidal attention hex tiling to a DiT model.

    Uses attn1_patch to inject wrapped K/V entries for boundary patches,
    so boundary patches attend to opposite-edge content.

    :param model_patcher: ComfyUI ModelPatcher instance
    :param settings: Tiling settings
    """
    from .toroidal_attention import HexToroidalAttentionPatch

    diff_model = model_patcher.model.diffusion_model

    if not hasattr(diff_model, 'pe_embedder'):
        raise ValueError(
            "Model does not have pe_embedder. "
            "Toroidal attention tiling requires a model with RoPE position embeddings."
        )

    patch = HexToroidalAttentionPatch(settings, diff_model.pe_embedder)
    model_patcher.set_model_attn1_patch(patch)
