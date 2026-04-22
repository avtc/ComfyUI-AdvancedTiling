"""
Raylight integration for DiT tiling.

Provides AdvancedTilingRay node that works with raylight's RAY_ACTORS type,
applying the same post_input hook tiling via @ray_patch on distributed workers.
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
        Uses @ray_patch to inject post_input hook on each distributed worker.
        """

        # pylint: disable=invalid-name

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "settings": ("ADVANCED_TILING_SETTINGS",),
                    "ray_actors": ("RAY_ACTORS",),
                    "padding": (
                        "INT",
                        {"default": 16, "min": 4, "max": 64, "step": 1},
                    ),
                },
            }

        RETURN_TYPES = ("RAY_ACTORS",)
        RETURN_NAMES = ("ray_actors",)
        FUNCTION = "run"
        CATEGORY = "conditioning"

        @ray_patch
        def patch(self, model, settings, padding=16):
            patch_dit_model(model, settings, padding)

        def run(self, settings, ray_actors, padding=16):
            self.patch(ray_actors, settings, padding)
            return (ray_actors,)
