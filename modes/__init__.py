"""
Collection of tiling modes
"""

# ruff: noqa: E402
# pylint: disable=wrong-import-position
# This is to solve circular imports


class Settings:
    """
    For representing tiling settings
    """

    def __init__(self, mode, rotation, scale=1.0, min_margin=0, divisible_by=1):
        self.mode = mode
        self.tiling_fn = modes[mode]
        self.rotation = rotation
        self.scale = scale
        self.min_margin = min_margin
        self.divisible_by = divisible_by

    def __hash__(self):
        return hash((self.mode, self.rotation, self.scale, self.min_margin, self.divisible_by))


from .hex import hex_tiling
from .none import none_tiling
from .rect import rect_tiling

modes = {
    "None": none_tiling,
    "Hexagon": hex_tiling,
    "Rectangular": rect_tiling,
}


__all__ = ["modes", "Settings"]
