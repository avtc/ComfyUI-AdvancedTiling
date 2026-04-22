"""
Raylight integration for DiT tiling.

Provides AdvancedTilingRay node that works with raylight's RAY_ACTORS type,
applying the same per-layer tiling via @ray_patch on distributed workers.
Only registers if raylight is installed.
"""

try:
    from raylight.comfy_extra_dist.ray_patch_decorator import ray_patch
    HAS_RAYLIGHT = True
except ImportError:
    HAS_RAYLIGHT = False

if HAS_RAYLIGHT:
    from .dit_tiling import patch_dit_model
    from .modes import Settings

    class AdvancedTilingRay:
        """
        Applies tiling to a raylight RAY_ACTORS model.
        Uses @ray_patch to inject per-layer hooks on each distributed worker.
        """

        # pylint: disable=invalid-name

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "settings": ("ADVANCED_TILING_SETTINGS",),
                    "ray_actors": ("RAY_ACTORS",),
                    "dit_padding": (
                        "INT",
                        {
                            "default": 16,
                            "min": 0,
                            "max": 128,
                            "step": 4,
                        },
                    ),
                },
            }

        RETURN_TYPES = ("RAY_ACTORS",)
        RETURN_NAMES = ("ray_actors",)
        FUNCTION = "run"
        CATEGORY = "conditioning"

        @ray_patch
        def patch(self, model, settings, dit_padding):
            patch_dit_model(model, settings, dit_padding)

        def run(self, settings, ray_actors, dit_padding=16):
            self.patch(ray_actors, settings, dit_padding)
            return (ray_actors,)
