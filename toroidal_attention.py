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


# ---------------------------------------------------------------------------
# Lumina / NextDiT toroidal attention
# ---------------------------------------------------------------------------

class LuminaAttentionWrapper:
    """Wraps JointAttention.forward() to inject boundary K/V with synthetic RoPE.

    Lumina applies RoPE inside JointAttention.forward(), so there is no
    external attn1_patch hook.  This wrapper intercepts the forward call,
    runs QKV + RoPE normally, then injects extra K/V entries for boundary
    patches with position-correct synthetic freqs_cis computed via RoPE
    shift composition:  R(virtual) = R(source) @ R(shift).

    Activated only when ``transformer_options["tiling_img_shape"]`` is set
    (by the model wrapper), so the monkey-patch is a no-op for unrelated
    model uses.
    """

    def __init__(self, attn_module, rope_embedder, patch_size, pad_tokens_multiple, settings=None):
        self.attn = attn_module
        self.rope_embedder = rope_embedder
        self.axes_dim = rope_embedder.axes_dim
        self.theta = rope_embedder.theta
        self.patch_size = patch_size
        self.pad_tokens_multiple = pad_tokens_multiple
        self.settings = settings

        self._initialized = False
        self._source_idx = None
        self._n_extra = 0
        self._shift_rope_h = None
        self._shift_rope_w = None
        self._axis_splits = (self.axes_dim[0] // 2, self.axes_dim[1] // 2)

        self._original_forward = attn_module.forward
        attn_module.forward = self._wrapped_forward

    def _initialize(self, h_patches, w_patches):
        from comfy.ldm.flux.math import rope as rope_fn

        if self.settings is not None and self.settings.mode == "Hexagon":
            b_idx, source_idx, off_h, off_w = _compute_hex_boundary_pairs(
                h_patches, w_patches, self.settings,
            )
        else:
            b_idx, source_idx, off_h, off_w = _compute_rect_boundary_pairs(
                h_patches, w_patches,
            )

        self._source_idx = source_idx
        self._n_extra = len(source_idx)

        if self._n_extra > 0:
            boundary_h = b_idx // w_patches
            boundary_w = b_idx % w_patches
            source_h = source_idx // w_patches
            source_w = source_idx % w_patches

            shift_h = (boundary_h.float() + off_h.float() - source_h.float()).unsqueeze(0)
            shift_w = (boundary_w.float() + off_w.float() - source_w.float()).unsqueeze(0)

            shift_rope_h = rope_fn(shift_h, self.axes_dim[1], self.theta)
            shift_rope_w = rope_fn(shift_w, self.axes_dim[2], self.theta)

            # Shape: (1, n_extra, 1, axis_dim//2, 2, 2) — add singleton for
            # broadcasting with the (batch, n_extra, 1, ...) source freqs.
            self._shift_rope_h = shift_rope_h.unsqueeze(2)
            self._shift_rope_w = shift_rope_w.unsqueeze(2)

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

        if self._n_extra == 0:
            return self._original_forward(x, x_mask, freqs_cis, transformer_options)

        from comfy.ldm.flux.math import apply_rope, apply_rope1
        from comfy.ldm.modules.attention import optimized_attention_masked

        # Image token offset in the full (text+image) sequence
        n_img = h_patches * w_patches
        if self.pad_tokens_multiple is not None:
            n_img_padded = -(-n_img // self.pad_tokens_multiple) * self.pad_tokens_multiple
        else:
            n_img_padded = n_img
        cap_size_0 = freqs_cis.shape[1] - n_img_padded

        # --- QKV projection + QK norm (same as original) ---
        bsz, seqlen, _ = x.shape
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

        # --- Apply RoPE to full sequence ---
        xq, xk = apply_rope(xq, xk, freqs_cis)

        # --- Extract boundary source K/V ---
        src_global = self._source_idx.to(xk.device) + cap_size_0
        extra_k = xk[:, src_global, :, :]
        extra_v = xv[:, src_global, :, :]

        # --- Synthetic freqs_cis via RoPE shift composition ---
        source_global = self._source_idx.to(freqs_cis.device) + cap_size_0
        source_freqs = freqs_cis[:, source_global, :, :, :, :]

        a0, a1 = self._axis_splits
        src_ax0 = source_freqs[:, :, :, :a0, :, :]
        src_ax1 = source_freqs[:, :, :, a0:a0 + a1, :, :]
        src_ax2 = source_freqs[:, :, :, a0 + a1:, :, :]

        shift_h = self._shift_rope_h.to(device=src_ax1.device, dtype=src_ax1.dtype)
        shift_w = self._shift_rope_w.to(device=src_ax2.device, dtype=src_ax2.dtype)

        syn_ax1 = torch.matmul(src_ax1, shift_h)
        syn_ax2 = torch.matmul(src_ax2, shift_w)
        syn_freqs = torch.cat([src_ax0, syn_ax1, syn_ax2], dim=3)

        # --- Apply RoPE to extra K with synthetic positions ---
        extra_k = apply_rope1(extra_k, syn_freqs)

        # --- Concatenate extra K/V ---
        xk = torch.cat([xk, extra_k], dim=1)
        xv = torch.cat([xv, extra_v], dim=1)

        # --- GQA expansion + attention (same as original) ---
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


class LuminaWastePatch:
    """double_block patch that replaces hex waste-position tokens with their
    mapped source content after each transformer block, preventing them from
    polluting boundary attention in subsequent layers."""

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
