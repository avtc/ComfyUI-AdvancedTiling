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

    def __init__(self, mode, rotation, blend_amount=0.0, blend_width=64,
                 latent_wrapping=True, position_fix=True, toroidal_attention=True):
        self.mode = mode
        self.tiling_fn = modes[mode]
        self.rotation = rotation
        self.blend_amount = blend_amount
        self.blend_width = blend_width
        self.latent_wrapping = latent_wrapping
        self.position_fix = position_fix
        self.toroidal_attention = toroidal_attention

    def __hash__(self):
        return hash((self.mode, self.rotation, self.blend_amount, self.blend_width,
                     self.latent_wrapping, self.position_fix, self.toroidal_attention))


from .hex import hex_tiling
from .none import none_tiling

modes = {
    "None": none_tiling,
    "Hexagon": hex_tiling,
}


__all__ = ["modes", "Settings"]
