"""
Hex grid preview: arranges up to 7 images in a hexagonal grid layout
with each image cropped to its hex shape.
"""

import math
import torch
import numpy as np

from .modes import Settings
from .modes.hex import hex_to_pixel, get_matrix
from .modes.hex_mask import NEIGHBOR_DIRECTIONS


# Axial coordinate offsets for 6 neighbors (matching NEIGHBOR_DIRECTIONS order)
NEIGHBOR_OFFSETS = {
    "E":  (1, 0),
    "NE": (1, -1),
    "NW": (0, -1),
    "W":  (-1, 0),
    "SW": (-1, 1),
    "SE": (0, 1),
}


def _create_hex_mask(hex_size: int, settings: Settings) -> torch.Tensor:
    """
    Create a binary hexagonal mask.

    Uses hex_tiling_vectorized with a virtual square image of size 2*hex_size
    so the resulting hex fills the mask exactly.

    :return: Bool tensor of shape (2*hex_size, 2*hex_size)
    """
    from .modes.hex import hex_tiling_vectorized

    dim = 2 * hex_size

    mapped_x, mapped_y = hex_tiling_vectorized(dim, dim, settings)

    xs = np.arange(dim, dtype=np.int64)[np.newaxis, :]
    ys = np.arange(dim, dtype=np.int64)[:, np.newaxis]

    is_inside = (mapped_x == xs) & (mapped_y == ys)
    return torch.from_numpy(is_inside)


class AdvancedTilingHexGridPreview:
    """
    Preview node that arranges up to 7 images in a hexagonal grid.
    Each image is masked to its hex shape and placed at the correct
    position on a black canvas.
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
            },
            "optional": {
                f"neighbor_{d}": ("IMAGE", {"tooltip": f"{d} neighbor tile image"})
                for d in NEIGHBOR_DIRECTIONS
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image"

    def run(self, center_image, hex_scale, rotation, gap, **kwargs):
        settings = Settings("Hexagon", rotation, hex_scale)

        B, tile_h, tile_w, C = center_image.shape
        hex_size = max(1, round(min(tile_w, tile_h) // 2 * hex_scale))

        # Create the hex mask: shape (2*hex_size, 2*hex_size)
        hex_mask = _create_hex_mask(hex_size, settings)

        # Compute pixel positions for each hex center relative to (0,0)
        positions = {"center": (0, 0)}
        for direction in NEIGHBOR_DIRECTIONS:
            q, r = NEIGHBOR_OFFSETS[direction]
            px, py = hex_to_pixel((q, r), hex_size, settings)
            positions[direction] = (px, py)

        # Add gap: push non-center positions radially outward
        if gap > 0:
            new_positions = {}
            for k, (px, py) in positions.items():
                if k == "center":
                    new_positions[k] = (px, py)
                else:
                    dist = math.sqrt(px * px + py * py)
                    if dist > 0:
                        new_positions[k] = (round(px + gap * px / dist), round(py + gap * py / dist))
                    else:
                        new_positions[k] = (px, py)
            positions = new_positions

        # Collect all images with their positions (skip mismatched sizes)
        images = [("center", center_image)]
        for direction in NEIGHBOR_DIRECTIONS:
            key = f"neighbor_{direction}"
            if key in kwargs and kwargs[key] is not None:
                nimg = kwargs[key]
                if nimg.shape[1] != tile_h or nimg.shape[2] != tile_w:
                    continue
                images.append((direction, nimg))

        # Compute output canvas size
        all_positions = [positions[name] for name, _ in images]
        min_px = min(px - tile_w // 2 for px, _ in all_positions)
        max_px = max(px + tile_w // 2 for px, _ in all_positions)
        min_py = min(py - tile_h // 2 for _, py in all_positions)
        max_py = max(py + tile_h // 2 for _, py in all_positions)

        canvas_w = max_px - min_px
        canvas_h = max_py - min_py

        canvas = torch.zeros(B, canvas_h, canvas_w, C, dtype=center_image.dtype)

        mask_h, mask_w = hex_mask.shape

        for name, img in images:
            px, py = positions[name]
            # Top-left corner of this tile in canvas coords
            x0 = px - tile_w // 2 - min_px
            y0 = py - tile_h // 2 - min_py

            # Center the hex mask on the tile
            mask_x0 = (tile_w - mask_w) // 2
            mask_y0 = (tile_h - mask_h) // 2

            # Build a full tile-sized mask, centered
            full_mask = torch.zeros(tile_h, tile_w, dtype=torch.float32)
            full_mask[mask_y0:mask_y0 + mask_h, mask_x0:mask_x0 + mask_w] = hex_mask.float()

            # Apply mask and paste
            mask_expanded = full_mask.unsqueeze(0).unsqueeze(-1)  # (1, tile_h, tile_w, 1)
            masked_img = img * mask_expanded
            canvas[:, y0:y0 + tile_h, x0:x0 + tile_w, :] += masked_img

        return (canvas,)
