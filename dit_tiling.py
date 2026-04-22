"""
DiT model tiling via post_input hook patching.

Works with any DiT model that supports the post_input transformer_options hook,
including Qwen Image 2512 via both standard ComfyUI and raylight FSDP/USP.

The padded sequence layout is: [original patches] + [padding patches].
This ensures the model's output crop ([:, :num_embeds]) correctly extracts
the original patches, while padding patches provide seamless wrap-around context.
"""

import torch
import torch.nn as nn
from torch.nn import Conv2d

from .modes import Settings
from .modes.hex import hex_patch_tiling


def _has_conv2d(model: nn.Module) -> bool:
    """Check if the model has any Conv2d layers (UNet vs DiT)."""
    return any(isinstance(m, Conv2d) for m in model.modules())


def _build_padding(
    h_patches: int,
    w_patches: int,
    padding: int,
    settings: Settings,
    device: torch.device,
    dtype: torch.dtype,
    h_positions: torch.Tensor,
    w_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute source indices and img_ids for padding patches.

    Iterates over the 2D padded grid, skipping original-region positions.
    For each padding position, finds the source via hex tiling and computes
    the actual 2D spatial position (extrapolated from the original img_ids pattern).

    :param h_patches: Patch rows in original grid
    :param w_patches: Patch columns in original grid
    :param padding: Padding amount on each side
    :param settings: Tiling settings
    :param device: Target device for tensors
    :param dtype: Target dtype for img_ids
    :param h_positions: Sorted unique h positions from original img_ids
    :param w_positions: Sorted unique w positions from original img_ids
    :return: (source_indices, padding_img_ids) where source_indices is LongTensor
             of flat indices into the original grid, and padding_img_ids is (1, P, 3)
    """
    padded_h = h_patches + 2 * padding
    padded_w = w_patches + 2 * padding

    h_min = h_positions[0].item()
    w_min = w_positions[0].item()
    h_step = (h_positions[-1] - h_positions[0]).item() / max(h_patches - 1, 1)
    w_step = (w_positions[-1] - w_positions[0]).item() / max(w_patches - 1, 1)

    sources = []
    positions = []

    for ph in range(padded_h):
        for pw in range(padded_w):
            oh = ph - padding
            ow = pw - padding

            if 0 <= oh < h_patches and 0 <= ow < w_patches:
                continue

            src_h, src_w = hex_patch_tiling(
                ph, pw,
                h_patches, w_patches,
                padded_h, padded_w,
                settings,
            )
            src_h = (src_h - padding) % h_patches
            src_w = (src_w - padding) % w_patches
            sources.append(src_h * w_patches + src_w)

            actual_h = h_min + oh * h_step
            actual_w = w_min + ow * w_step
            positions.append([0.0, actual_h, actual_w])

    sources_tensor = torch.tensor(sources, dtype=torch.long, device=device)
    positions_tensor = torch.tensor(positions, dtype=dtype, device=device).unsqueeze(0)

    return sources_tensor, positions_tensor


def create_dit_tiling_patch(settings: Settings, padding: int):
    """
    Create a post_input patch function for DiT tiling.

    The returned function conforms to the post_input hook signature:
      fn({"img": ..., "txt": ..., "img_ids": ..., "txt_ids": ..., "transformer_options": ...}) -> dict

    Layout: [original patches (N)] + [padding patches (P)]
    The model's output crop ([:, :N]) extracts original patches correctly.

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
            _cache[cache_key] = _build_padding(
                h_patches, w_patches, padding, settings,
                hidden_states.device, img_ids.dtype,
                h_positions, w_positions,
            )

        pad_sources, pad_img_ids = _cache[cache_key]

        # Gather padding hidden states from source positions in original grid
        pad_hidden = hidden_states[:, pad_sources, :]

        # Concatenate: original patches first, then padding patches
        padded_hidden = torch.cat([hidden_states, pad_hidden], dim=1)
        padded_img_ids = torch.cat([img_ids, pad_img_ids.expand(bsz, -1, -1)], dim=1)

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
