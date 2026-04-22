"""
DiT model tiling via per-layer hidden state replacement.

Works with any DiT model that supports the post_input and double_block
transformer_options hooks, including Qwen Image via both standard ComfyUI
and raylight FSDP/USP.

Instead of adding padding patches to the sequence (which the model's attention
ignores), this approach keeps the same sequence length and replaces the hidden
states of "waste" patches (outside the hexagonal tile boundary) with content
from the opposite edge after every transformer block. This is the direct DiT
analog of Conv2d tiling — same image size, wrapping enforced at every layer.

The post_input hook captures the 2D patch layout from img_ids and precomputes
the waste-to-source mapping. The double_block hook applies the mapping after
each transformer block.
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
    Compute waste-to-source index mapping for hex tiling.

    Uses hex_tiling with original_size == padded_size to determine which
    patches are inside the hexagon (identity mapping) vs outside (waste).
    Waste patches map to their hex-wrapped source on the opposite edge.

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


def create_dit_tiling_patch(settings: Settings):
    """
    Create per-layer tiling patches for DiT models.

    Returns (post_input_fn, double_block_fn) — two hook functions:
    - post_input_fn: captures 2D layout from img_ids, precomputes mapping
    - double_block_fn: replaces waste patches after each transformer block

    :param settings: Tiling settings
    :return: (post_input_fn, double_block_fn)
    """
    _cache = {}

    def post_input(data: dict) -> dict:
        img_ids = data["img_ids"]
        transformer_options = data["transformer_options"]

        h_positions = img_ids[0, :, 1].unique().sort()[0]
        w_positions = img_ids[0, :, 2].unique().sort()[0]
        h_patches = h_positions.shape[0]
        w_patches = w_positions.shape[0]

        cache_key = (h_patches, w_patches, hash(settings))
        if cache_key not in _cache:
            _cache[cache_key] = _build_waste_mapping(
                h_patches, w_patches, settings,
            )

        transformer_options["tiling_waste_idx"] = _cache[cache_key][0]
        transformer_options["tiling_source_idx"] = _cache[cache_key][1]

        return data

    def double_block(data: dict) -> dict:
        img = data["img"]
        transformer_options = data["transformer_options"]

        waste_idx = transformer_options.get("tiling_waste_idx")
        source_idx = transformer_options.get("tiling_source_idx")

        if waste_idx is not None and source_idx is not None:
            img[:, waste_idx] = img[:, source_idx]

        return data

    return post_input, double_block


def patch_dit_model(model_patcher, settings: Settings):
    """
    Apply DiT tiling patch to a ComfyUI ModelPatcher.

    Uses post_input hook to capture layout and double_block hook to replace
    waste patches after each transformer block.

    :param model_patcher: ComfyUI ModelPatcher instance
    :param settings: Tiling settings
    """
    post_input_fn, double_block_fn = create_dit_tiling_patch(settings)
    model_patcher.set_model_post_input_patch(post_input_fn)
    model_patcher.set_model_double_block_patch(double_block_fn)
