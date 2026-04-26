"""
Hex grid preview: arranges up to 7 images in a hexagonal grid layout
using the same hex coordinate system as hex_tiling_vectorized for
pixel-perfect alignment.
"""

import math
import logging

import torch
import numpy as np

from .modes import Settings
from .modes.hex_mask import NEIGHBOR_DIRECTIONS

logger = logging.getLogger("ComfyUI-AdvancedTiling")

# Axial coordinate offsets for 6 neighbors (matching NEIGHBOR_DIRECTIONS order)
NEIGHBOR_OFFSETS = {
    "E":  (1, 0),
    "NE": (1, -1),
    "NW": (0, -1),
    "W":  (-1, 0),
    "SW": (-1, 1),
    "SE": (0, 1),
}


def _compute_hex_grid(
    canvas_w: int, canvas_h: int, hex_size: int,
    tile_w: int, tile_h: int, settings: Settings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute hex cell membership for each canvas pixel.

    Uses the same hex coordinate math as hex_tiling_vectorized but on a
    multi-cell canvas without modulo wrapping.

    :return: (cell_q, cell_r, tile_x, tile_y)
        cell_q, cell_r: int64 (canvas_h, canvas_w) — hex cell axial coords
        tile_x, tile_y: int64 (canvas_h, canvas_w) — pixel coords within tile image
    """
    from .modes.hex import get_matrix, get_inverse_matrix

    matrix = get_matrix(settings)
    inv_matrix = get_inverse_matrix(settings)

    ma, mb = float(matrix[0, 0]), float(matrix[0, 1])
    mc, md = float(matrix[1, 0]), float(matrix[1, 1])
    ia, ib = float(inv_matrix[0, 0]), float(inv_matrix[0, 1])
    ic, id_ = float(inv_matrix[1, 0]), float(inv_matrix[1, 1])

    ys, xs = np.meshgrid(
        np.arange(canvas_h, dtype=np.float64),
        np.arange(canvas_w, dtype=np.float64),
        indexing='ij',
    )
    xs -= canvas_w / 2.0
    ys -= canvas_h / 2.0

    # pixel_to_hex — same as hex_tiling_vectorized
    q = (ia * xs + ib * ys) / hex_size
    r = (ic * xs + id_ * ys) / hex_size

    # cube_round (vectorized) — matches hex_tiling_vectorized exactly
    s = -q - r
    rq = np.rint(q)
    rr = np.rint(r)
    rs = np.rint(s)

    q_diff = np.abs(rq - q)
    r_diff = np.abs(rr - r)
    s_diff = np.abs(rs - s)

    cond_q = (q_diff > r_diff) & (q_diff > s_diff)
    cond_r = ~cond_q & (r_diff > s_diff)

    # Must match hex_tiling_vectorized: rq modified before rr uses it
    rq = np.where(cond_q, -rr - rs, rq)
    rr = np.where(cond_r, -rq - rs, rr)

    cell_q = rq.astype(np.int64)
    cell_r = rr.astype(np.int64)

    # Fractional offset from hex center
    frac_q = q - rq
    frac_r = r - rr

    # hex_to_pixel for fractional offset — same as hex_tiling_vectorized
    px = hex_size * (ma * frac_q + mb * frac_r)
    py = hex_size * (mc * frac_q + md * frac_r)

    # Map to tile image coordinates (center hex mask at tile center)
    tile_x = np.rint(px).astype(np.int64) + tile_w // 2
    tile_y = np.rint(py).astype(np.int64) + tile_h // 2

    return cell_q, cell_r, tile_x, tile_y


class AdvancedTilingHexGridPreview:
    """
    Preview node that arranges up to 7 images in a hexagonal grid.

    Uses the same hex coordinate system as hex_tiling_vectorized for
    pixel-perfect alignment between adjacent hexes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "center_image": ("IMAGE", {"tooltip": "Center hex tile"}),
                "hex_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Hex size as fraction of tile half-size. 1.0 = hex fills the tile.",
                    },
                ),
                "rotation": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 360.0,
                        "step": 1.0,
                        "tooltip": "Hex rotation in degrees.",
                    },
                ),
                "gap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 50,
                        "tooltip": "Pixel gap between hexagons.",
                    },
                ),
                "enable_vae_decode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "VAE encode/decode each tile before display.",
                    },
                ),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "VAE for encode/decode (required if enable_vae_decode is True)"}),
                **{
                    f"neighbor_{d}": ("IMAGE", {"tooltip": f"{d} neighbor tile image"})
                    for d in NEIGHBOR_DIRECTIONS
                },
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image"

    def run(
        self, center_image, hex_scale, rotation, gap,
        enable_vae_decode=False, vae=None, **kwargs,
    ):
        settings = Settings("Hexagon", rotation, hex_scale)

        # VAE round-trip if enabled
        if enable_vae_decode and vae is not None:
            center_image = self._vae_roundtrip(center_image, vae)
            for direction in NEIGHBOR_DIRECTIONS:
                key = f"neighbor_{direction}"
                if key in kwargs and kwargs[key] is not None:
                    kwargs[key] = self._vae_roundtrip(kwargs[key], vae)

        B, tile_h, tile_w, C = center_image.shape
        hex_size = max(1, round(min(tile_w, tile_h) // 2 * hex_scale))
        dim = 2 * hex_size

        logger.info(f"[HexGridPreview] tile=({tile_w}x{tile_h}), "
                     f"hex_size={hex_size}, dim={dim}, scale={hex_scale}")

        # Build inside mask using hex_tiling_vectorized (same as hex_inpaint)
        from .modes.hex import hex_tiling_vectorized
        mapped_x, mapped_y = hex_tiling_vectorized(dim, dim, settings)
        mxs = np.arange(dim, dtype=np.int64)[np.newaxis, :]
        mys = np.arange(dim, dtype=np.int64)[:, np.newaxis]
        inside_mask = (mapped_x == mxs) & (mapped_y == mys)  # (dim, dim) bool

        inside_count = int(inside_mask.sum())
        logger.info(f"[HexGridPreview] inside_mask shape={inside_mask.shape}, "
                     f"inside_pixels={inside_count}/{dim*dim}")

        # Apply gap via erosion
        if gap > 0:
            from .modes.hex_mask import _erode_mask
            inside_mask = _erode_mask(torch.from_numpy(inside_mask), gap).numpy()
            logger.info(f"[HexGridPreview] after erosion gap={gap}: "
                         f"inside_pixels={int(inside_mask.sum())}")

        # Canvas large enough for center + 6 neighbors
        canvas_radius = int(math.ceil(3 * hex_size))
        canvas_w = 2 * canvas_radius
        canvas_h = 2 * canvas_radius

        logger.info(f"[HexGridPreview] canvas=({canvas_w}x{canvas_h}), "
                     f"canvas_radius={canvas_radius}")

        # Compute hex cell membership for each canvas pixel
        cell_q, cell_r, tile_x, tile_y = _compute_hex_grid(
            canvas_w, canvas_h, hex_size, tile_w, tile_h, settings,
        )

        # DEBUG: check center pixel
        cy_c, cx_c = canvas_h // 2, canvas_w // 2
        logger.info(f"[HexGridPreview] center pixel ({cx_c},{cy_c}): "
                     f"cell=({cell_q[cy_c, cx_c]},{cell_r[cy_c, cx_c]}), "
                     f"tile=({tile_x[cy_c, cx_c]},{tile_y[cy_c, cx_c]})")

        # Determine inside status for each canvas pixel using the same mask
        # tile_x/y are in tile image coords; convert to hex mask coords for lookup
        hx = tile_x - (tile_w // 2 - hex_size)  # hex mask x
        hy = tile_y - (tile_h // 2 - hex_size)  # hex mask y

        logger.info(f"[HexGridPreview] center hx/hy=({hx[cy_c, cx_c]},{hy[cy_c, cx_c]}), "
                     f"tile_w//2={tile_w//2}, hex_size={hex_size}")

        hx_c = np.clip(hx, 0, dim - 1)
        hy_c = np.clip(hy, 0, dim - 1)
        is_inside = inside_mask[hy_c, hx_c]
        is_valid = (hx >= 0) & (hx < dim) & (hy >= 0) & (hy < dim)
        is_inside = is_inside & is_valid

        total_inside = int(is_inside.sum())
        unique_cells = set(zip(cell_q[is_inside].tolist(), cell_r[is_inside].tolist()))
        logger.info(f"[HexGridPreview] total_inside={total_inside}/{canvas_h*canvas_w}, "
                     f"unique_cells={sorted(unique_cells)}")

        # Build tile lookup: (q, r) -> image
        tile_lookup = {(0, 0): center_image}
        for direction in NEIGHBOR_DIRECTIONS:
            key = f"neighbor_{direction}"
            if key in kwargs and kwargs[key] is not None:
                nimg = kwargs[key]
                if nimg.shape[1] == tile_h and nimg.shape[2] == tile_w:
                    q, r = NEIGHBOR_OFFSETS[direction]
                    tile_lookup[(q, r)] = nimg

        logger.info(f"[HexGridPreview] tile_lookup keys={list(tile_lookup.keys())}")

        # Render canvas
        canvas = torch.zeros(B, canvas_h, canvas_w, C, dtype=center_image.dtype)

        cell_q_t = torch.from_numpy(cell_q)
        cell_r_t = torch.from_numpy(cell_r)
        tile_x_t = torch.from_numpy(np.clip(tile_x, 0, tile_w - 1))
        tile_y_t = torch.from_numpy(np.clip(tile_y, 0, tile_h - 1))
        is_inside_t = torch.from_numpy(is_inside)

        for (q, r), img in tile_lookup.items():
            cell_match = (cell_q_t == q) & (cell_r_t == r)
            pixel_mask = cell_match & is_inside_t
            n_cell = int(cell_match.sum())
            n_masked = int(pixel_mask.sum())
            logger.info(f"[HexGridPreview] cell ({q},{r}): "
                         f"cell_match={n_cell}, inside={n_masked}")
            if not pixel_mask.any():
                continue
            cy, cx = pixel_mask.nonzero(as_tuple=True)
            ty = tile_y_t[cy, cx]
            tx = tile_x_t[cy, cx]
            logger.info(f"[HexGridPreview]   cy range=[{cy.min().item()},{cy.max().item()}], "
                         f"cx range=[{cx.min().item()},{cx.max().item()}], "
                         f"ty range=[{ty.min().item()},{ty.max().item()}], "
                         f"tx range=[{tx.min().item()},{tx.max().item()}]")
            logger.info(f"[HexGridPreview]   canvas device={canvas.device}, dtype={canvas.dtype}, "
                         f"img device={img.device}, dtype={img.dtype}")
            logger.info(f"[HexGridPreview]   img sample at center: {img[0, 664, 664, :].tolist()}")
            canvas[:, cy, cx, :] = img[:, ty, tx, :]
            logger.info(f"[HexGridPreview]   canvas sum after write: {canvas.sum().item():.4f}")

        logger.info(f"[HexGridPreview] total canvas sum={canvas.sum().item():.4f}, "
                     f"nonzero={int((canvas > 0).sum())}")

        # Crop to content bounds
        content = canvas[0].sum(dim=-1) > 0  # (H, W) bool
        row_has_content = content.any(dim=1)  # (H,)
        col_has_content = content.any(dim=0)  # (W,)
        if row_has_content.any() and col_has_content.any():
            row_idx = row_has_content.nonzero(as_tuple=True)[0]
            col_idx = col_has_content.nonzero(as_tuple=True)[0]
            r0, r1 = row_idx[0].item(), row_idx[-1].item() + 1
            c0, c1 = col_idx[0].item(), col_idx[-1].item() + 1
            canvas = canvas[:, r0:r1, c0:c1, :]
        else:
            logger.warning("[HexGridPreview] NO content in canvas!")

        logger.info(f"[HexGridPreview] final canvas={canvas.shape}")
        return (canvas,)

    @staticmethod
    def _vae_roundtrip(image: torch.Tensor, vae) -> torch.Tensor:
        latent = vae.encode(image)
        decoded = vae.decode(latent)
        if decoded.ndim == 5:
            decoded = decoded.squeeze(1)
        return decoded
