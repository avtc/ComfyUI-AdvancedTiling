"""
Collection of tiling modes
"""


MODE_NAMES = ["None", "Hexagon", "Rectangular"]

# Sentinel values: scale=0 means auto, min_margin=-1 means auto.
# After _resolve_auto(), these are replaced with concrete values.
AUTO_SCALE = 0.0
AUTO_MARGIN = -1


class Settings:
    """Raw tiling settings from user input. May contain sentinel values.

    Sentinel values:
      scale=0.0      -> auto (resolved based on model type)
      min_margin=-1  -> auto (resolved based on model type)

    Use _resolve_auto() to produce a ResolvedSettings with all values concrete
    and resolution-dependent properties pre-computed.
    """

    __slots__ = ("mode", "rotation", "scale", "min_margin", "divisible_by")

    def __init__(self, mode, rotation, scale, min_margin, divisible_by):
        self.mode = mode
        self.rotation = rotation
        self.scale = scale
        self.min_margin = min_margin
        self.divisible_by = divisible_by

    def _resolve_auto(self, is_conv2d: bool, vae_factor: int, patch_size: int,
                      img_w: int, img_h: int) -> "ResolvedSettings":
        """Resolve auto-sentinel values and pre-compute all resolution-dependent properties.

        Margin pipeline (floor semantics):
          base_margin = W * (1 - scale) / 2
          margin = max(base_margin, min_margin_img)
          work = W - 2 * margin
          work = work // divisible_by * divisible_by
          margin = (W - work) / 2

        Returns a ResolvedSettings with working area and margins at image,
        latent, and patch resolutions — no downstream computation needed.
        """
        scale = self.scale
        if scale == AUTO_SCALE:
            scale = 1.0 if (is_conv2d or self.mode == "Hexagon") else 7 / 8

        min_margin = self.min_margin
        if min_margin == AUTO_MARGIN:
            min_margin = 4 if (not is_conv2d and self.mode == "Rectangular") else 0

        patch_vae_factor = vae_factor * patch_size

        if self.mode == "Rectangular":
            base_margin_w = img_w * (1 - scale) / 2
            base_margin_h = img_h * (1 - scale) / 2
            min_margin_img = min_margin * vae_factor
            margin_w = max(base_margin_w, min_margin_img)
            margin_h = max(base_margin_h, min_margin_img)
            work_w = img_w - 2 * margin_w
            work_h = img_h - 2 * margin_h
            if self.divisible_by > 1:
                work_w = work_w // self.divisible_by * self.divisible_by
                work_h = work_h // self.divisible_by * self.divisible_by
            margin_w = (img_w - work_w) / 2
            margin_h = (img_h - work_h) / 2

            work_lat_w = work_w / vae_factor
            work_lat_h = work_h / vae_factor
            margin_lat_w = margin_w / vae_factor
            margin_lat_h = margin_h / vae_factor
            work_patch_w = work_w / patch_vae_factor
            work_patch_h = work_h / patch_vae_factor
            margin_patch_w = margin_w / patch_vae_factor
            margin_patch_h = margin_h / patch_vae_factor
            hex_size_img = hex_size_lat = hex_size_patch = 0.0

        elif self.mode == "Hexagon":
            min_dim = min(img_w, img_h)
            base_margin = min_dim / 2 * (1 - scale)
            min_margin_img = min_margin * vae_factor
            margin = max(base_margin, min_margin_img)
            hex_size = min_dim / 2 - margin
            if self.divisible_by > 1:
                hex_height = 2 * hex_size
                hex_height = hex_height // self.divisible_by * self.divisible_by
                hex_size = hex_height / 2
            hex_size_img = hex_size
            hex_size_lat = hex_size / vae_factor
            hex_size_patch = hex_size / patch_vae_factor

            # work = bounding square of the hex, margin = gap to image edge
            work_w = work_h = 2 * hex_size
            margin_w = (img_w - work_w) / 2
            margin_h = (img_h - work_h) / 2
            work_lat_w = work_lat_h = 2 * hex_size_lat
            margin_lat_w = margin_w / vae_factor
            margin_lat_h = margin_h / vae_factor
            work_patch_w = work_patch_h = 2 * hex_size_patch
            margin_patch_w = margin_w / patch_vae_factor
            margin_patch_h = margin_h / patch_vae_factor

        else:  # "None"
            work_w = work_h = margin_w = margin_h = 0.0
            work_lat_w = work_lat_h = margin_lat_w = margin_lat_h = 0.0
            work_patch_w = work_patch_h = margin_patch_w = margin_patch_h = 0.0
            hex_size_img = hex_size_lat = hex_size_patch = 0.0

        return ResolvedSettings(
            self.mode, self.rotation,
            work_w, work_h, margin_w, margin_h,
            work_lat_w, work_lat_h, margin_lat_w, margin_lat_h,
            work_patch_w, work_patch_h, margin_patch_w, margin_patch_h,
            hex_size_img, hex_size_lat, hex_size_patch,
            patch_vae_factor,
            img_w, img_h,
        )

    def __eq__(self, other):
        if not isinstance(other, Settings):
            return NotImplemented
        return (self.mode, self.rotation, self.scale, self.min_margin,
                self.divisible_by) == (
                    other.mode, other.rotation, other.scale,
                    other.min_margin, other.divisible_by)

    def __hash__(self):
        return hash((self.mode, self.rotation, self.scale, self.min_margin,
                     self.divisible_by))


class ResolvedSettings:
    """Fully resolved tiling settings with all resolution-dependent properties.

    Produced by Settings._resolve_auto(). Guaranteed to have concrete values
    for all fields — no sentinel values, no None properties.

    work/margin fields: bounding box of the working area and gap to image edge.
      Rectangular: the working area IS the rectangle.
      Hexagon:     the working area is the bounding square of the hex.
      None:        all zeros (handled by early return in consumers).

    hex_size fields: hex radius (center to vertex). Populated for Hexagon, zero otherwise.

    Invariant: work_img_w + 2 * margin_img_w == img_w (and same for h).
    """

    __slots__ = (
        "mode", "rotation",
        "work_img_w", "work_img_h", "margin_img_w", "margin_img_h",
        "work_lat_w", "work_lat_h", "margin_lat_w", "margin_lat_h",
        "work_patch_w", "work_patch_h", "margin_patch_w", "margin_patch_h",
        "hex_size_img", "hex_size_lat", "hex_size_patch",
        "patch_vae_factor",
        "img_w", "img_h",
    )

    def __init__(self, mode, rotation,
                 work_img_w, work_img_h, margin_img_w, margin_img_h,
                 work_lat_w, work_lat_h, margin_lat_w, margin_lat_h,
                 work_patch_w, work_patch_h, margin_patch_w, margin_patch_h,
                 hex_size_img, hex_size_lat, hex_size_patch,
                 patch_vae_factor,
                 img_w, img_h):
        self.mode = mode
        self.rotation = rotation
        self.work_img_w = work_img_w
        self.work_img_h = work_img_h
        self.margin_img_w = margin_img_w
        self.margin_img_h = margin_img_h
        self.work_lat_w = work_lat_w
        self.work_lat_h = work_lat_h
        self.margin_lat_w = margin_lat_w
        self.margin_lat_h = margin_lat_h
        self.work_patch_w = work_patch_w
        self.work_patch_h = work_patch_h
        self.margin_patch_w = margin_patch_w
        self.margin_patch_h = margin_patch_h
        self.hex_size_img = hex_size_img
        self.hex_size_lat = hex_size_lat
        self.hex_size_patch = hex_size_patch
        self.patch_vae_factor = patch_vae_factor
        self.img_w = img_w
        self.img_h = img_h

    def __eq__(self, other):
        if not isinstance(other, ResolvedSettings):
            return NotImplemented
        return (self.mode, self.rotation,
                self.work_img_w, self.work_img_h, self.margin_img_w, self.margin_img_h,
                self.work_lat_w, self.work_lat_h, self.margin_lat_w, self.margin_lat_h,
                self.work_patch_w, self.work_patch_h, self.margin_patch_w, self.margin_patch_h,
                self.hex_size_img, self.hex_size_lat, self.hex_size_patch,
                self.patch_vae_factor,
                self.img_w, self.img_h) == (
                    other.mode, other.rotation,
                    other.work_img_w, other.work_img_h, other.margin_img_w, other.margin_img_h,
                    other.work_lat_w, other.work_lat_h, other.margin_lat_w, other.margin_lat_h,
                    other.work_patch_w, other.work_patch_h, other.margin_patch_w, other.margin_patch_h,
                    other.hex_size_img, other.hex_size_lat, other.hex_size_patch,
                    other.patch_vae_factor,
                    other.img_w, other.img_h)

    def __hash__(self):
        return hash((self.mode, self.rotation,
                     self.work_img_w, self.work_img_h, self.margin_img_w, self.margin_img_h,
                     self.work_lat_w, self.work_lat_h, self.margin_lat_w, self.margin_lat_h,
                     self.work_patch_w, self.work_patch_h, self.margin_patch_w, self.margin_patch_h,
                     self.hex_size_img, self.hex_size_lat, self.hex_size_patch,
                     self.patch_vae_factor,
                     self.img_w, self.img_h))


__all__ = ["MODE_NAMES", "Settings", "ResolvedSettings", "AUTO_SCALE", "AUTO_MARGIN"]
