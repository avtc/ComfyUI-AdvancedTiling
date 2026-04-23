"""
Extended toroidal attention for hex neighbor context injection.

For each boundary patch, injects K/V from the appropriate neighbor tile
based on which direction the boundary faces, instead of self-wrapping.
"""

import torch

from .modes import Settings
from .modes.hex_mask import NEIGHBOR_DIRECTIONS
from .toroidal_attention import _BaseToroidalAttentionPatch, _factorize


def _compute_hex_boundary_pairs_with_direction(
    h_patches: int,
    w_patches: int,
    settings: Settings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Like _compute_hex_boundary_pairs but also records direction index (0-5).

    Direction is determined by the offset (dh, dw) of the boundary patch
    relative to the wrapped source.

    :return: (boundary_idx, source_idx, off_h, off_w, direction_idx) as LongTensors
    """
    from .modes.hex import hex_tiling
    from .modes.hex_mask import _pixel_angle_from_center, _angle_to_direction

    boundary_idx = []
    source_idx = []
    offsets_h = []
    offsets_w = []
    directions = []

    for h in range(h_patches):
        for w in range(w_patches):
            src_w, src_h = hex_tiling(
                w, h, (w_patches, h_patches), (w_patches, h_patches), settings
            )
            if src_w != w or src_h != h:
                continue

            for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nh, nw = h + dh, w + dw
                n_src_w, n_src_h = hex_tiling(
                    nw, nh, (w_patches, h_patches), (w_patches, h_patches), settings
                )

                if n_src_w != nw or n_src_h != nh:
                    angle = _pixel_angle_from_center(w, h, w_patches, h_patches)
                    direction = _angle_to_direction(angle)

                    boundary_idx.append(h * w_patches + w)
                    source_idx.append(n_src_h * w_patches + n_src_w)
                    offsets_h.append(dh)
                    offsets_w.append(dw)
                    directions.append(direction)

    if not boundary_idx:
        empty = torch.tensor([], dtype=torch.long)
        return (empty, empty.clone(), empty.clone(), empty.clone(), empty.clone())

    return (
        torch.tensor(boundary_idx, dtype=torch.long),
        torch.tensor(source_idx, dtype=torch.long),
        torch.tensor(offsets_h, dtype=torch.long),
        torch.tensor(offsets_w, dtype=torch.long),
        torch.tensor(directions, dtype=torch.long),
    )


class HexNeighborAttentionPatch(_BaseToroidalAttentionPatch):
    """
    Attention patch that injects K/V from neighbor tiles at boundary patches.

    When neighbors are provided in the latent (Hybrid/Masked modes), this
    works identically to HexToroidalAttentionPatch since neighbors are already
    in the K/V tensor. For Attention-Only mode, neighbor K/V must be pre-loaded.
    """

    def __init__(self, settings: Settings, pe_embedder):
        super().__init__(pe_embedder)
        self.settings = settings
        self._directions = None

    def _compute_boundary_pairs(self, h_patches, w_patches):
        result = _compute_hex_boundary_pairs_with_direction(
            h_patches, w_patches, self.settings
        )
        self._directions = result[4]
        return result[:4]
