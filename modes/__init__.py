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

    def __init__(self, mode, rotation, scale=1.0, *,
                 rope_fix=False, timestep_decay=False, boundary_blend=False):
        self.mode = mode
        self.tiling_fn = modes[mode]
        self.rotation = rotation
        self.scale = scale
        self.rope_fix = rope_fix
        self.timestep_decay = timestep_decay
        self.boundary_blend = boundary_blend

    def __hash__(self):
        return hash((self.mode, self.rotation, self.scale,
                     self.rope_fix, self.timestep_decay, self.boundary_blend))


from .hex import hex_tiling
from .none import none_tiling
from .rect import rect_tiling

modes = {
    "None": none_tiling,
    "Hexagon": hex_tiling,
    "Rectangular": rect_tiling,
}


__all__ = ["modes", "Settings"]
