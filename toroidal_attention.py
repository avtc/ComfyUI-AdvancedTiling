"""
Toroidal attention patches for hex and rectangular tiling in DiT models.

Injects wrapped-neighbor K/V entries into every attention layer via attn1_patch,
so boundary patches structurally "see" opposite-edge content as spatially adjacent.
Analogous to how Conv2d circular padding works for UNet models.
"""

import torch

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


# ---------------------------------------------------------------------------
# Hex toroidal attention
# ---------------------------------------------------------------------------

_boundary_cache: dict = {}


def _compute_hex_boundary_pairs(
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
            src_w, src_h = hex_tiling(
                w, h,
                (w_patches, h_patches),
                (w_patches, h_patches),
                settings,
            )
            if src_w != w or src_h != h:
                continue

            for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nh, nw = h + dh, w + dw

                n_src_w, n_src_h = hex_tiling(
                    nw, nh,
                    (w_patches, h_patches),
                    (w_patches, h_patches),
                    settings,
                )

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


# ---------------------------------------------------------------------------
# Rectangular toroidal attention
# ---------------------------------------------------------------------------

_rect_boundary_cache: dict = {}


def _compute_rect_boundary_pairs(
    h_patches: int,
    w_patches: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute rectangular boundary pairs for toroidal wrapping.

    For each patch on the edge of the grid, records the boundary patch index,
    the wrapped source index from the opposite edge, and the direction offset.

    Cached by (h_patches, w_patches).

    :param h_patches: Number of patch rows
    :param w_patches: Number of patch columns
    :return: (boundary_idx, source_idx, off_h, off_w) as LongTensors
    """
    cache_key = (h_patches, w_patches)
    if cache_key in _rect_boundary_cache:
        return _rect_boundary_cache[cache_key]

    boundary_idx = []
    source_idx = []
    offsets_h = []
    offsets_w = []

    for h in range(h_patches):
        for w in range(w_patches):
            for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nh, nw = h + dh, w + dw

                if nh < 0 or nh >= h_patches or nw < 0 or nw >= w_patches:
                    wrapped_h = nh % h_patches
                    wrapped_w = nw % w_patches

                    boundary_idx.append(h * w_patches + w)
                    source_idx.append(wrapped_h * w_patches + wrapped_w)
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

    _rect_boundary_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Shared base for attention patches
# ---------------------------------------------------------------------------

class _BaseToroidalAttentionPatch:
    """Shared logic for hex and rectangular toroidal attention patches."""

    def __init__(self, pe_embedder):
        self.pe_embedder = pe_embedder
        self._initialized = False
        self._boundary_idx = None
        self._source_idx = None
        self._n_extra = 0
        self._synthetic_pe = None

    def _compute_boundary_pairs(self, h_patches, w_patches):
        raise NotImplementedError

    def _initialize(self, n_img: int):
        h_patches, w_patches = _factorize(n_img)

        boundary_idx, source_idx, off_h, off_w = self._compute_boundary_pairs(
            h_patches, w_patches,
        )

        self._boundary_idx = boundary_idx
        self._source_idx = source_idx
        self._n_extra = len(boundary_idx)

        if self._n_extra == 0:
            self._initialized = True
            return

        boundary_h = boundary_idx // w_patches
        boundary_w = boundary_idx % w_patches

        h_center = h_patches // 2
        w_center = w_patches // 2

        n_axes = len(self.pe_embedder.axes_dim)
        syn_ids = torch.zeros(1, self._n_extra, n_axes, dtype=torch.float32)
        syn_ids[0, :, 0] = 0
        syn_ids[0, :, 1] = (boundary_h + off_h).float() - h_center
        syn_ids[0, :, 2] = (boundary_w + off_w).float() - w_center

        with torch.no_grad():
            self._synthetic_pe = self.pe_embedder(syn_ids)

        self._initialized = True

    def __call__(self, q, k, v, pe=None, attn_mask=None, extra_options=None):
        if extra_options is None or "img_slice" not in extra_options:
            return {}

        if attn_mask is not None:
            return {}

        if not self._initialized:
            img_slice = extra_options["img_slice"]
            n_img = img_slice[1] - img_slice[0]
            self._initialize(n_img)

        if self._n_extra == 0:
            return {}

        n_txt = extra_options["img_slice"][0]

        src_joint_idx = self._source_idx.to(k.device) + n_txt
        extra_k = k[:, :, src_joint_idx, :]
        extra_v = v[:, :, src_joint_idx, :]

        new_k = torch.cat([k, extra_k], dim=2)
        new_v = torch.cat([v, extra_v], dim=2)

        synthetic_pe = self._synthetic_pe.to(device=pe.device, dtype=pe.dtype)
        expand_shape = list(pe.shape)
        expand_shape[2] = synthetic_pe.shape[2]
        new_pe = torch.cat([
            pe,
            synthetic_pe.expand(expand_shape),
        ], dim=2)

        return {
            "k": new_k,
            "v": new_v,
            "pe": new_pe,
        }


class HexToroidalAttentionPatch(_BaseToroidalAttentionPatch):
    """attn1_patch for hex tiling: injects wrapped K/V for hex boundary patches."""

    def __init__(self, settings: Settings, pe_embedder):
        super().__init__(pe_embedder)
        self.settings = settings

    def _compute_boundary_pairs(self, h_patches, w_patches):
        return _compute_hex_boundary_pairs(h_patches, w_patches, self.settings)


class RectToroidalAttentionPatch(_BaseToroidalAttentionPatch):
    """attn1_patch for rectangular tiling: injects wrapped K/V from opposite edges."""

    def _compute_boundary_pairs(self, h_patches, w_patches):
        return _compute_rect_boundary_pairs(h_patches, w_patches)
