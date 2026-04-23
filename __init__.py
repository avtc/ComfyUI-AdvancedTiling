"""
ComfyUI Node entry point
"""

# pylint: disable=invalid-name

from .advanced_tiling import (
    AdvancedTilingSettings,
    AdvancedTiling,
    AdvancedTilingVAEDecode,
)
from .hex_inpaint import AdvancedTilingHexInpaint

NODE_CLASS_MAPPINGS = {
    "AdvancedTilingSettings": AdvancedTilingSettings,
    "AdvancedTiling": AdvancedTiling,
    "AdvancedTilingVAEDecode": AdvancedTilingVAEDecode,
    "AdvancedTilingHexInpaint": AdvancedTilingHexInpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedTilingSettings": "Advanced Tiling Settings",
    "AdvancedTiling": "Advanced Tiling",
    "AdvancedTilingVAEDecode": "Advanced Tiling VAE Decode",
    "AdvancedTilingHexInpaint": "Advanced Tiling Hex Inpaint",
}

try:
    from .ray_tiling import AdvancedTilingRay, HAS_RAYLIGHT
    if HAS_RAYLIGHT:
        NODE_CLASS_MAPPINGS["AdvancedTilingRay"] = AdvancedTilingRay
        NODE_DISPLAY_NAME_MAPPINGS["AdvancedTilingRay"] = "Advanced Tiling (Raylight)"
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
