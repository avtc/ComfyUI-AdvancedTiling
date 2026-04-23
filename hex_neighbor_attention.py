"""
Hex neighbor attention for inpainting border transitions.

Injects K/V from waste region patches (which hold neighbor content after
compositing) at boundary positions, reinforcing the neighbor context signal
during denoising.
"""

import torch

from .modes import Settings
from .modes.hex_mask import NEIGHBOR_DIRECTIONS
from .toroidal_attention import _BaseToroidalAttentionPatch, _factorize


def _compute_hex_boundary_to_waste_pairs(
    h_patches: int,
    w_patches: int,
    settings: Settings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find boundary-inside patches adjacent to waste-outside patches.

    For each patch inside the hex that has a 4-connected neighbor in the
    waste region, record the boundary patch index and the waste patch index.
    K/V from the waste patch (neighbor content) will be injected at the
    boundary patch position.

    :return: (boundary_idx, waste_idx, off_h, off_w) as LongTensors
    """
    from .modes.hex import hex_tiling

    boundary_idx = []
    waste_idx = []
    offsets_h = []
    offsets_w = []

    for h in range(h_patches):
        for w in range(w_patches):
            src_w, src_h = hex_tiling(
                w, h, (w_patches, h_patches), (w_patches, h_patches), settings
            )
            # Skip waste patches — only process inside-hex patches
            if src_w != w or src_h != h:
                continue

            for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nh, nw = h + dh, w + dw

                # Bounds check — out-of-bounds is always waste
                if nh < 0 or nh >= h_patches or nw < 0 or nw >= w_patches:
                    continue

                n_src_w, n_src_h = hex_tiling(
                    nw, nh, (w_patches, h_patches), (w_patches, h_patches), settings
                )

                # If neighbor maps elsewhere, it's a waste patch
                if n_src_w != nw or n_src_h != nh:
                    boundary_idx.append(h * w_patches + w)
                    waste_idx.append(nh * w_patches + nw)
                    offsets_h.append(dh)
                    offsets_w.append(dw)

    if not boundary_idx:
        empty = torch.tensor([], dtype=torch.long)
        return (empty, empty.clone(), empty.clone(), empty.clone())

    return (
        torch.tensor(boundary_idx, dtype=torch.long),
        torch.tensor(waste_idx, dtype=torch.long),
        torch.tensor(offsets_h, dtype=torch.long),
        torch.tensor(offsets_w, dtype=torch.long),
    )


class HexNeighborAttentionPatch(_BaseToroidalAttentionPatch):
    """
    Attention patch for hex inpainting that injects K/V from waste region
    patches (neighbor content) at boundary positions.

    In Hybrid/Masked modes, the waste region contains composited neighbor
    content. This patch reinforces that context by injecting extra K/V from
    waste patches at each attention layer.
    """

    def __init__(self, settings: Settings, pe_embedder):
        super().__init__(pe_embedder)
        self.settings = settings

    def _compute_boundary_pairs(self, h_patches, w_patches):
        return _compute_hex_boundary_to_waste_pairs(
            h_patches, w_patches, self.settings
        )
