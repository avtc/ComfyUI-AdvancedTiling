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
    margin_h: int = 0,
    margin_w: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute rectangular boundary pairs for toroidal wrapping.

    For each patch on the edge of the working rectangle (inside the margins),
    records the boundary patch index, the wrapped source index from the
    opposite edge of the working rectangle, and the direction offset.

    Cached by (h_patches, w_patches, margin_h, margin_w).

    :param h_patches: Number of patch rows (full grid)
    :param w_patches: Number of patch columns (full grid)
    :param margin_h: Margin rows on each side
    :param margin_w: Margin columns on each side
    :return: (boundary_idx, source_idx, off_h, off_w) as LongTensors
    """
    cache_key = (h_patches, w_patches, margin_h, margin_w)
    if cache_key in _rect_boundary_cache:
        return _rect_boundary_cache[cache_key]

    work_h = h_patches - 2 * margin_h
    work_w = w_patches - 2 * margin_w

    boundary_idx = []
    source_idx = []
    offsets_h = []
    offsets_w = []

    for h in range(margin_h, margin_h + work_h):
        for w in range(margin_w, margin_w + work_w):
            for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nh, nw = h + dh, w + dw

                if nh < margin_h or nh >= margin_h + work_h or nw < margin_w or nw >= margin_w + work_w:
                    wrapped_h = margin_h + (nh - margin_h) % work_h
                    wrapped_w = margin_w + (nw - margin_w) % work_w

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

    def __init__(self, pe_embedder, scale=1.0):
        self.pe_embedder = pe_embedder
        self.scale = scale
        self._initialized = False
        self._boundary_idx = None
        self._source_idx = None
        self._n_extra = 0
        self._synthetic_pe = None

    def _compute_boundary_pairs(self, h_patches, w_patches, margin_h=0, margin_w=0):
        raise NotImplementedError

    @staticmethod
    def _compute_margins(h_patches, w_patches, scale):
        if scale >= 1.0:
            return 0, 0
        work_h = max(1, round(h_patches * scale))
        work_w = max(1, round(w_patches * scale))
        return (h_patches - work_h) // 2, (w_patches - work_w) // 2

    def _initialize(self, n_img: int):
        h_patches, w_patches = _factorize(n_img)
        margin_h, margin_w = self._compute_margins(h_patches, w_patches, self.scale)

        boundary_idx, source_idx, off_h, off_w = self._compute_boundary_pairs(
            h_patches, w_patches, margin_h, margin_w,
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
        super().__init__(pe_embedder, scale=settings.scale)
        self.settings = settings

    def _compute_boundary_pairs(self, h_patches, w_patches, margin_h=0, margin_w=0):
        return _compute_hex_boundary_pairs(h_patches, w_patches, self.settings)


class RectToroidalAttentionPatch(_BaseToroidalAttentionPatch):
    """attn1_patch for rectangular tiling: injects wrapped K/V from opposite edges."""

    def __init__(self, pe_embedder, scale=1.0):
        super().__init__(pe_embedder, scale=scale)

    def _compute_boundary_pairs(self, h_patches, w_patches, margin_h=0, margin_w=0):
        return _compute_rect_boundary_pairs(h_patches, w_patches, margin_h, margin_w)


# ---------------------------------------------------------------------------
# Lumina / NextDiT attention position correction
# ---------------------------------------------------------------------------

class LuminaAttentionWrapper:
    """Wraps JointAttention.forward() to correct waste token position encoding.

    The root cause of boundary artifacts: waste/margin tokens have content from
    the opposite edge (via latent wrapping) but position encoding for their
    actual margin position. This mismatch confuses attention — working tokens
    near the boundary attend to waste tokens and get contaminated by the
    position-content conflict, producing noise that spreads inward.

    Fix: before apply_rope, replace freqs_cis for waste tokens with their
    source position's freqs_cis. This makes waste tokens' K have the correct
    position encoding for their content, so working tokens get proper toroidal
    context without contamination.

    Unlike the previous K/V injection approach (which added extra tokens at
    virtual adjacent positions with very strong attention signal), this modifies
    existing waste tokens in-place. The attention signal is moderate (distance
    = opposite edge), not overwhelming.

    Activated only when transformer_options["tiling_img_shape"] is set.
    """

    def __init__(self, attn_module, patch_size, pad_tokens_multiple, settings):
        self.attn = attn_module
        self.patch_size = patch_size
        self.pad_tokens_multiple = pad_tokens_multiple
        self.settings = settings

        self._initialized = False
        self._waste_local = None
        self._source_local = None

        self._original_forward = attn_module.forward
        attn_module.forward = self._wrapped_forward

    def _initialize(self, h_patches, w_patches):
        waste_indices = []
        source_indices = []

        if self.settings.mode == "Hexagon":
            for h in range(h_patches):
                for w in range(w_patches):
                    src_w, src_h = hex_tiling(
                        w, h,
                        (w_patches, h_patches),
                        (w_patches, h_patches),
                        self.settings,
                    )
                    if src_w != w or src_h != h:
                        waste_indices.append(h * w_patches + w)
                        source_indices.append(src_h * w_patches + src_w)
        else:
            scale = self.settings.scale
            if scale < 1.0:
                work_h = max(1, round(h_patches * scale))
                work_w = max(1, round(w_patches * scale))
                margin_h = (h_patches - work_h) // 2
                margin_w = (w_patches - work_w) // 2
                for h in range(h_patches):
                    for w in range(w_patches):
                        if (h < margin_h or h >= margin_h + work_h
                                or w < margin_w or w >= margin_w + work_w):
                            waste_indices.append(h * w_patches + w)
                            src_h = margin_h + (h - margin_h) % work_h
                            src_w = margin_w + (w - margin_w) % work_w
                            source_indices.append(src_h * w_patches + src_w)

        if waste_indices:
            self._waste_local = torch.tensor(waste_indices, dtype=torch.long)
            self._source_local = torch.tensor(source_indices, dtype=torch.long)

        self._initialized = True

    def _wrapped_forward(self, x, x_mask, freqs_cis, transformer_options={}):
        img_shape = transformer_options.get("tiling_img_shape")
        if img_shape is None or x_mask is not None:
            return self._original_forward(x, x_mask, freqs_cis, transformer_options)

        H, W = img_shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        if not self._initialized:
            self._initialize(h_patches, w_patches)

        if self._waste_local is None:
            return self._original_forward(x, x_mask, freqs_cis, transformer_options)

        from comfy.ldm.flux.math import apply_rope
        from comfy.ldm.modules.attention import optimized_attention_masked

        bsz, seqlen, _ = x.shape

        # QKV projection + reshape + QK norm (same as original)
        xq, xk, xv = torch.split(
            self.attn.qkv(x),
            [
                self.attn.n_local_heads * self.attn.head_dim,
                self.attn.n_local_kv_heads * self.attn.head_dim,
                self.attn.n_local_kv_heads * self.attn.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.attn.n_local_heads, self.attn.head_dim)
        xk = xk.view(bsz, seqlen, self.attn.n_local_kv_heads, self.attn.head_dim)
        xv = xv.view(bsz, seqlen, self.attn.n_local_kv_heads, self.attn.head_dim)
        xq = self.attn.q_norm(xq)
        xk = self.attn.k_norm(xk)

        # Replace freqs_cis for waste tokens with their source position's freqs_cis
        n_img = h_patches * w_patches
        if self.pad_tokens_multiple is not None:
            n_img_padded = -(-n_img // self.pad_tokens_multiple) * self.pad_tokens_multiple
        else:
            n_img_padded = n_img
        cap_size = freqs_cis.shape[1] - n_img_padded

        waste_global = self._waste_local.to(freqs_cis.device) + cap_size
        source_global = self._source_local.to(freqs_cis.device) + cap_size

        modified_freqs = freqs_cis.clone()
        modified_freqs[:, waste_global] = freqs_cis[:, source_global]

        # Apply RoPE with corrected freqs_cis
        xq, xk = apply_rope(xq, xk, modified_freqs)

        # GQA expansion + attention (same as original)
        n_rep = self.attn.n_local_heads // self.attn.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        output = optimized_attention_masked(
            xq.movedim(1, 2), xk.movedim(1, 2), xv.movedim(1, 2),
            self.attn.n_local_heads, x_mask, skip_reshape=True,
            transformer_options=transformer_options,
        )
        return self.attn.out(output)


# ---------------------------------------------------------------------------
# Lumina waste/margin token reset patch
# ---------------------------------------------------------------------------

class LuminaWastePatch:
    """double_block patch that replaces waste/margin tokens with their
    mapped source content after each transformer block, preventing garbage
    accumulation from polluting attention for working-area patches.

    For Hexagon mode: uses hex_tiling to identify waste positions (outside hex)
    and their hex source positions.
    For Rectangular mode: identifies margin positions (outside working rectangle)
    and maps them to the opposite edge of the working rectangle.
    """

    def __init__(self, patch_size: int, settings: Settings):
        self.patch_size = patch_size
        self.settings = settings
        self._initialized = False
        self._waste_idx = None
        self._waste_source_idx = None

    def _initialize(self, x: torch.Tensor):
        _, _, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        waste_indices = []
        waste_source_indices = []

        if self.settings.mode == "Hexagon":
            for h in range(h_patches):
                for w in range(w_patches):
                    src_w, src_h = hex_tiling(
                        w, h,
                        (w_patches, h_patches),
                        (w_patches, h_patches),
                        self.settings,
                    )
                    if src_w != w or src_h != h:
                        waste_indices.append(h * w_patches + w)
                        waste_source_indices.append(src_h * w_patches + src_w)
        else:  # Rectangular
            scale = self.settings.scale
            if scale >= 1.0:
                work_h, work_w = h_patches, w_patches
            else:
                work_h = max(1, round(h_patches * scale))
                work_w = max(1, round(w_patches * scale))
            margin_h = (h_patches - work_h) // 2
            margin_w = (w_patches - work_w) // 2

            for h in range(h_patches):
                for w in range(w_patches):
                    is_margin = (
                        h < margin_h or h >= margin_h + work_h
                        or w < margin_w or w >= margin_w + work_w
                    )
                    if is_margin:
                        waste_indices.append(h * w_patches + w)
                        src_h = margin_h + (h - margin_h) % work_h
                        src_w = margin_w + (w - margin_w) % work_w
                        waste_source_indices.append(src_h * w_patches + src_w)

        if waste_indices:
            self._waste_idx = torch.tensor(waste_indices, dtype=torch.long)
            self._waste_source_idx = torch.tensor(waste_source_indices, dtype=torch.long)

        self._initialized = True

    def __call__(self, data: dict) -> dict:
        if not self._initialized:
            self._initialize(data["x"])

        if self._waste_idx is None:
            return {}

        img = data["img"]
        img[:, self._waste_idx, :] = img[:, self._waste_source_idx, :]
        return {"img": img}
