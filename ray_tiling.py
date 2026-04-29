"""
Raylight integration for DiT tiling.

Provides AdvancedTilingRay node that works with raylight's RAY_ACTORS type,
applying toroidal attention tiling on distributed workers.

Uses manual Ray remote calls instead of @ray_patch to avoid serialization
issues with custom node packages that have non-standard import paths
(e.g. hyphens in directory names that prevent normal Python imports).
"""

try:
    import raylight.comfy_extra_dist.ray_patch_decorator  # noqa: F401
    HAS_RAYLIGHT = True
except ImportError:
    HAS_RAYLIGHT = False

if HAS_RAYLIGHT:
    import ray

    class AdvancedTilingRay:
        """
        Applies tiling to a raylight RAY_ACTORS model.
        Dispatches per-worker patching via Ray remote calls.
        """

        # pylint: disable=invalid-name

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "settings": ("ADVANCED_TILING_SETTINGS",),
                    "ray_actors": ("RAY_ACTORS",),
                },
            }

        RETURN_TYPES = ("RAY_ACTORS",)
        RETURN_NAMES = ("ray_actors",)
        FUNCTION = "run"
        CATEGORY = "conditioning"

        def run(self, settings, ray_actors):
            # Raylight only supports DiT models (is_conv2d=False)
            settings = settings._resolve_auto(is_conv2d=False)

            # Defined inside run() so cloudpickle serializes it as a nested
            # function (by value) instead of by module reference.  Module-level
            # functions get serialized by reference, which requires importing
            # the module by name on the worker -- but the module name is the
            # filesystem path (with a hyphen), causing ModuleNotFoundError.
            def _patch(model, mode, rotation, scale, min_margin, divisible_by):
                import importlib.util
                import os
                import sys

                _node_dir = os.path.dirname(os.path.abspath(__file__))
                _pkg = "ComfyUI_AdvancedTiling"

                if _pkg not in sys.modules:
                    _init = os.path.join(_node_dir, "__init__.py")
                    spec = importlib.util.spec_from_file_location(
                        _pkg, _init,
                        submodule_search_locations=[_node_dir],
                    )
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[_pkg] = mod
                    spec.loader.exec_module(mod)

                from ComfyUI_AdvancedTiling.dit_tiling import patch_dit_model
                from ComfyUI_AdvancedTiling.modes import Settings

                patch_dit_model(model, Settings(mode, rotation, scale, min_margin, divisible_by))
                return model

            gpu_workers = ray_actors["workers"]
            futures = [
                actor.model_function_runner.remote(
                    _patch, settings.mode, settings.rotation, settings.scale, settings.min_margin, settings.divisible_by
                )
                for actor in gpu_workers
            ]
            ray.get(futures)
            return (ray_actors,)
