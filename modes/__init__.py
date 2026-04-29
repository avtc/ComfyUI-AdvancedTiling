"""
Collection of tiling modes
"""


MODE_NAMES = ["None", "Hexagon", "Rectangular"]


class Settings:
    """
    For representing tiling settings.

    Immutable by convention — use _resolve_auto() to produce a new resolved
    instance instead of mutating in place.
    """

    __slots__ = ("mode", "rotation", "scale", "min_margin", "divisible_by")

    def __init__(self, mode, rotation, scale=1.0, min_margin=0, divisible_by=1):
        self.mode = mode
        self.rotation = rotation
        self.scale = scale
        self.min_margin = min_margin
        self.divisible_by = divisible_by

    def _resolve_auto(self, is_conv2d: bool) -> "Settings":
        """Return a new Settings with auto-sentinel values resolved.

        scale=0.0 is auto: Conv2d/Hexagon → 1.0, DiT Rectangular → 7/8.
        min_margin=-1 is auto: 4 for DiT Rectangular, 0 otherwise.
        """
        scale = self.scale
        if scale == 0.0:
            scale = 1.0 if (is_conv2d or self.mode == "Hexagon") else 7 / 8

        min_margin = self.min_margin
        if min_margin == -1:
            min_margin = 4 if (not is_conv2d and self.mode == "Rectangular") else 0

        return Settings(self.mode, self.rotation, scale, min_margin, self.divisible_by)

    def __eq__(self, other):
        if not isinstance(other, Settings):
            return NotImplemented
        return (self.mode, self.rotation, self.scale, self.min_margin,
                self.divisible_by) == (other.mode, other.rotation, other.scale,
                                       other.min_margin, other.divisible_by)

    def __hash__(self):
        return hash((self.mode, self.rotation, self.scale, self.min_margin, self.divisible_by))


__all__ = ["MODE_NAMES", "Settings"]
