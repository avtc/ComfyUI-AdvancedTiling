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

    def __init__(self, mode, rotation, blend_amount=0.0, blend_width=64):
        self.mode = mode
        self.tiling_fn = modes[mode]
        self.rotation = rotation
        self.blend_amount = blend_amount
        self.blend_width = blend_width

    def __hash__(self):
        return hash((self.mode, self.rotation, self.blend_amount, self.blend_width))


from .hex import hex_tiling
from .none import none_tiling

modes = {
    "None": none_tiling,
    "Hexagon": hex_tiling,
}


__all__ = ["modes", "Settings"]
