"""
DiT model tiling via post_input hook patching.

Works with any DiT model that supports the post_input transformer_options hook,
including Qwen Image 2512 via both standard ComfyUI and raylight FSDP/USP.
"""

import torch
import torch.nn as nn
from torch.nn import Conv2d

from .modes import Settings
from .modes.hex import hex_patch_tiling


def _has_conv2d(model: nn.Module) -> bool:
    """Check if the model has any Conv2d layers (UNet vs DiT)."""
    return any(isinstance(m, Conv2d) for m in model.modules())


def create_patch_mapping(
    h_patches: int,
    w_patches: int,
    padding: int,
    settings: Settings,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build a source-index mapping for padded patch positions.

    Returns a 1D LongTensor of length (h_patches + 2*padding) * (w_patches + 2*padding)
    where each element is the flat source index in the original (h_patches, w_patches) grid.
    Positions inside the original grid map to themselves.
    Positions in the padding region map via hex tiling remapping.

    :param h_patches: Number of patch rows in original grid
    :param w_patches: Number of patch columns in original grid
    :param padding: Number of padding patch rows/columns on each side
    :param settings: Tiling settings
    :return: LongTensor of flat source indices
    """
    padded_h = h_patches + 2 * padding
    padded_w = w_patches + 2 * padding
    mapping = torch.zeros(padded_h * padded_w, dtype=torch.long, device=device)

    for ph in range(padded_h):
        for pw in range(padded_w):
            flat_idx = ph * padded_w + pw
            oh = ph - padding
            ow = pw - padding

            if 0 <= oh < h_patches and 0 <= ow < w_patches:
                mapping[flat_idx] = oh * w_patches + ow
            else:
                src_h, src_w = hex_patch_tiling(
                    ph, pw,
                    h_patches, w_patches,
                    padded_h, padded_w,
                    settings,
                )
                src_h = src_h % h_patches
                src_w = src_w % w_patches
                mapping[flat_idx] = src_h * w_patches + src_w

    return mapping


def create_img_ids_mapping(
    h_patches: int,
    w_patches: int,
    padding: int,
    img_ids: torch.Tensor,
    settings: Settings,
) -> torch.Tensor:
    """
    Build padded img_ids where padding positions get the img_ids
    of their source position (so RoPE encodes them as adjacent to the opposite edge).

    :param h_patches: Original patch rows
    :param w_patches: Original patch columns
    :param padding: Padding amount
    :param img_ids: Original img_ids of shape (B, N, 3)
    :param settings: Tiling settings
    :return: Padded img_ids of shape (B, N + padding_patches, 3)
    """
    padded_h = h_patches + 2 * padding
    padded_w = w_patches + 2 * padding
    bsz = img_ids.shape[0]

    padded_img_ids = torch.zeros(bsz, padded_h * padded_w, 3,
                                 device=img_ids.device, dtype=img_ids.dtype)

    for ph in range(padded_h):
        for pw in range(padded_w):
            padded_flat = ph * padded_w + pw
            oh = ph - padding
            ow = pw - padding

            if 0 <= oh < h_patches and 0 <= ow < w_patches:
                src_flat = oh * w_patches + ow
            else:
                src_h, src_w = hex_patch_tiling(
                    ph, pw,
                    h_patches, w_patches,
                    padded_h, padded_w,
                    settings,
                )
                src_h = src_h % h_patches
                src_w = src_w % w_patches
                src_flat = src_h * w_patches + src_w

            padded_img_ids[:, padded_flat, :] = img_ids[:, src_flat, :]

    return padded_img_ids


def create_dit_tiling_patch(settings: Settings, padding: int):
    """
    Create a post_input patch function for DiT tiling.

    The returned function conforms to the post_input hook signature:
      fn({"img": ..., "txt": ..., "img_ids": ..., "txt_ids": ..., "transformer_options": ...}) -> dict

    :param settings: Tiling settings
    :param padding: Number of patches to pad on each side
    :return: Patch function
    """
    _cache = {}

    def tiling_post_input(data: dict) -> dict:
        hidden_states = data["img"]
        encoder_hidden_states = data["txt"]
        img_ids = data["img_ids"]
        txt_ids = data["txt_ids"]
        transformer_options = data["transformer_options"]

        bsz, num_patches, dim = hidden_states.shape

        h_positions = img_ids[0, :, 1].unique().sort()[0]
        w_positions = img_ids[0, :, 2].unique().sort()[0]
        h_patches = h_positions.shape[0]
        w_patches = w_positions.shape[0]

        if num_patches != h_patches * w_patches:
            return data

        cache_key = (h_patches, w_patches, padding, hash(settings))
        if cache_key not in _cache:
            patch_mapping = create_patch_mapping(
                h_patches, w_patches, padding, settings, device=hidden_states.device
            )
            img_ids_mapping = create_img_ids_mapping(
                h_patches, w_patches, padding, img_ids, settings
            )
            _cache[cache_key] = (patch_mapping, img_ids_mapping)

        patch_mapping, padded_img_ids = _cache[cache_key]
        padded_hidden = hidden_states[:, patch_mapping, :]

        return {
            "img": padded_hidden,
            "txt": encoder_hidden_states,
            "img_ids": padded_img_ids,
            "txt_ids": txt_ids,
            "transformer_options": transformer_options,
        }

    return tiling_post_input


def patch_dit_model(model_patcher, settings: Settings, padding: int = 16):
    """
    Apply DiT tiling patch to a ComfyUI ModelPatcher.

    Works with both standard ModelPatcher and FSDPModelPatcher.
    Uses set_model_post_input_patch() to inject the tiling hook.

    :param model_patcher: ComfyUI ModelPatcher instance
    :param settings: Tiling settings
    :param padding: Number of patches to pad on each side
    """
    patch_fn = create_dit_tiling_patch(settings, padding)
    model_patcher.set_model_post_input_patch(patch_fn)
