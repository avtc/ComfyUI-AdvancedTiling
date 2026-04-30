"""
Collection of tiling modes
"""


MODE_NAMES = ["None", "Hexagon", "Rectangular"]

# Sentinel values: scale=0 means auto, min_margin=-1 means auto.
# After _resolve_auto(), these are replaced with concrete values.
AUTO_SCALE = 0.0
AUTO_MARGIN = -1


class Settings:
    """
    For representing tiling settings.

    Immutable by convention — use _resolve_auto() to produce a new resolved
    instance instead of mutating in place.

    Sentinel values:
      scale=0.0      -> auto (resolved based on model type)
      min_margin=-1  -> auto (resolved based on model type)

    After _resolve_auto(), all values are concrete and _resolved=True.
    """

    __slots__ = ("mode", "rotation", "scale", "min_margin", "divisible_by", "_resolved")

    def __init__(self, mode, rotation, scale, min_margin, divisible_by):
        self.mode = mode
        self.rotation = rotation
        self.scale = scale
        self.min_margin = min_margin
        self.divisible_by = divisible_by
        self._resolved = (scale != AUTO_SCALE and min_margin != AUTO_MARGIN)

    @property
    def resolved(self) -> bool:
        return self._resolved

    def _resolve_auto(self, is_conv2d: bool) -> "Settings":
        """Return a new Settings with auto-sentinel values resolved.

        scale=0.0 is auto: Conv2d/Hexagon -> 1.0, DiT Rectangular -> 7/8.
        min_margin=-1 is auto: 4 for DiT Rectangular, 0 otherwise.
        """
        scale = self.scale
        if scale == AUTO_SCALE:
            scale = 1.0 if (is_conv2d or self.mode == "Hexagon") else 7 / 8

        min_margin = self.min_margin
        if min_margin == AUTO_MARGIN:
            min_margin = 4 if (not is_conv2d and self.mode == "Rectangular") else 0

        s = Settings(self.mode, self.rotation, scale, min_margin, self.divisible_by)
        s._resolved = True
        return s

    def __eq__(self, other):
        if not isinstance(other, Settings):
            return NotImplemented
        return (self.mode, self.rotation, self.scale, self.min_margin,
                self.divisible_by) == (other.mode, other.rotation, other.scale,
                                       other.min_margin, other.divisible_by)

    def __hash__(self):
        return hash((self.mode, self.rotation, self.scale, self.min_margin, self.divisible_by))


__all__ = ["MODE_NAMES", "Settings", "AUTO_SCALE", "AUTO_MARGIN"]
