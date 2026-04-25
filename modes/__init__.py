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

    def __init__(self, mode, rotation, scale=1.0,
                 lumina_kv_injection=True, lumina_v_dampen=False, lumina_boundary_blend=False):
        self.mode = mode
        self.tiling_fn = modes[mode]
        self.rotation = rotation
        self.scale = scale
        self.lumina_kv_injection = lumina_kv_injection
        self.lumina_v_dampen = lumina_v_dampen
        self.lumina_boundary_blend = lumina_boundary_blend

    def __hash__(self):
        return hash((self.mode, self.rotation, self.scale))


from .hex import hex_tiling
from .none import none_tiling
from .rect import rect_tiling

modes = {
    "None": none_tiling,
    "Hexagon": hex_tiling,
    "Rectangular": rect_tiling,
}


__all__ = ["modes", "Settings"]
