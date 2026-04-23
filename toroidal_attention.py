"""
Toroidal attention patch for hex tiling in DiT models.

Injects wrapped-neighbor K/V entries into every attention layer via attn1_patch,
so boundary patches structurally "see" opposite-edge content as spatially adjacent.
Analogous to how Conv2d circular padding works for UNet models.
"""

import torch
import functools

from .modes import Settings
from .modes.hex import hex_tiling


def _factorize(n: int) -> tuple[int, int]:
    """Factorize n into h * w, preferring square."""
    s = int(n ** 0.5)
    while s > 0:
        if n % s == 0:
            return s, n // s
        s -= 1
    return 1, n


_boundary_cache: dict = {}


def _compute_boundary_pairs(
    h_patches: int,
    w_patches: int,
    settings: Settings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute hex boundary neighbor relationships at patch granularity.

    For each patch inside the hex that has a 4-connected neighbor outside
    the hex, record the boundary patch index, the wrapped source index,
    and the direction offset.

    Cached by (h_patches, w_patches, hash(settings)).

    :param h_patches: Number of patch rows
    :param w_patches: Number of patch columns
    :param settings: Tiling settings
    :return: (boundary_idx, source_idx, off_h, off_w) as LongTensors
    """
    cache_key = (h_patches, w_patches, hash(settings))
    if cache_key in _boundary_cache:
        return _boundary_cache[cache_key]

    boundary_idx = []
    source_idx = []
    offsets_h = []
    offsets_w = []

    for h in range(h_patches):
        for w in range(w_patches):
            # Check if this patch is inside the hex (identity mapping)
            src_w, src_h = hex_tiling(
                w, h,
                (w_patches, h_patches),
                (w_patches, h_patches),
                settings,
            )
            if src_w != w or src_h != h:
                continue  # Outside hex

            # Check 4-connected neighbors
            for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nh, nw = h + dh, w + dw

                # Find where neighbor maps to via hex_tiling
                n_src_w, n_src_h = hex_tiling(
                    nw, nh,
                    (w_patches, h_patches),
                    (w_patches, h_patches),
                    settings,
                )

                # If neighbor maps elsewhere, it's outside the hex
                if n_src_w != nw or n_src_h != nh:
                    boundary_idx.append(h * w_patches + w)
                    source_idx.append(n_src_h * w_patches + n_src_w)
                    offsets_h.append(dh)
                    offsets_w.append(dw)

    if not boundary_idx:
        empty = torch.tensor([], dtype=torch.long)
        result = (empty, empty.clone(), empty.clone(), empty.clone())
    else:
        result = (
            torch.tensor(boundary_idx, dtype=torch.long),
            torch.tensor(source_idx, dtype=torch.long),
            torch.tensor(offsets_h, dtype=torch.long),
            torch.tensor(offsets_w, dtype=torch.long),
        )

    _boundary_cache[cache_key] = result
    return result


class HexToroidalAttentionPatch:
    """
    attn1_patch that injects wrapped K/V entries for hex boundary patches.

    On first call, lazily initializes by:
    1. Extracting image dimensions from extra_options["img_slice"]
    2. Computing boundary pairs via _compute_boundary_pairs
    3. Precomputing synthetic position embeddings using pe_embedder

    On each call, extends K/V with wrapped source entries and PE with
    synthetic positions, so boundary patches attend to opposite-edge
    content as if it were spatially adjacent.

    Q is not extended, so attention output has the original shape.
    apply_rope1's truncation handles the pe/Q length mismatch.
    """

    def __init__(self, settings: Settings, pe_embedder):
        """
        :param settings: Tiling settings
        :param pe_embedder: The model's EmbedND module for computing RoPE
        """
        self.settings = settings
        self.pe_embedder = pe_embedder
        self._initialized = False
        self._boundary_idx = None
        self._source_idx = None
        self._n_extra = 0
        self._synthetic_pe = None

    def _initialize(self, n_img: int):
        """Lazy init: compute boundary pairs and synthetic PEs."""
        h_patches, w_patches = _factorize(n_img)

        boundary_idx, source_idx, off_h, off_w = _compute_boundary_pairs(
            h_patches, w_patches, self.settings,
        )

        self._boundary_idx = boundary_idx
        self._source_idx = source_idx
        self._n_extra = len(boundary_idx)

        if self._n_extra == 0:
            self._initialized = True
            return

        # Compute synthetic position IDs for the extra K/V entries.
        # Each extra entry represents the wrapped content that should appear
        # adjacent to a boundary patch. Its position is set to the boundary
        # patch's position + offset (one step toward the outside).
        boundary_h = boundary_idx // w_patches
        boundary_w = boundary_idx % w_patches

        h_center = h_patches // 2
        w_center = w_patches // 2

        syn_ids = torch.zeros(1, self._n_extra, 3, dtype=torch.float32)
        syn_ids[0, :, 0] = 0  # t_index
        syn_ids[0, :, 1] = (boundary_h + off_h).float() - h_center
        syn_ids[0, :, 2] = (boundary_w + off_w).float() - w_center

        # Compute PE using the model's pe_embedder (same as model.py:518)
        with torch.no_grad():
            self._synthetic_pe = self.pe_embedder(syn_ids)
            # Shape: (1, 1, n_extra, 64, 2, 2)

        self._initialized = True

    def __call__(self, q, k, v, pe=None, attn_mask=None, extra_options=None):
        """
        attn1_patch callback. Extends K/V with wrapped boundary entries.

        Called with: (joint_query, joint_key, joint_value, pe, attn_mask, extra_options)
        Returns: dict with optional keys: k, v, pe, attn_mask
        """
        if extra_options is None or "img_slice" not in extra_options:
            return {}

        # When attn_mask is present, the model's local attn_mask variable
        # (model.py:173-177) can't be extended from this hook. Skip to avoid
        # shape mismatch. This only affects batched unequal-length prompts.
        if attn_mask is not None:
            return {}

        if not self._initialized:
            img_slice = extra_options["img_slice"]
            n_img = img_slice[1] - img_slice[0]
            self._initialize(n_img)

        if self._n_extra == 0:
            return {}

        n_txt = extra_options["img_slice"][0]

        # Extract source K/V from image portion (offset by n_txt for joint space)
        src_joint_idx = self._source_idx.to(k.device) + n_txt
        extra_k = k[:, :, src_joint_idx, :]  # (B, H, n_extra, d)
        extra_v = v[:, :, src_joint_idx, :]  # (B, H, n_extra, d)

        # Extend K/V with wrapped source entries
        new_k = torch.cat([k, extra_k], dim=2)
        new_v = torch.cat([v, extra_v], dim=2)

        # Extend PE with synthetic position embeddings.
        # apply_rope1(Q, extended_pe) -> pe truncated to Q length (math.py:34-35)
        # apply_rope1(K, extended_pe) -> full pe including synthetic entries
        synthetic_pe = self._synthetic_pe.to(device=pe.device, dtype=pe.dtype)
        new_pe = torch.cat([
            pe,
            synthetic_pe.expand(pe.shape[0], -1, -1, -1, -1, -1),
        ], dim=2)

        return {
            "k": new_k,
            "v": new_v,
            "pe": new_pe,
        }
