"""
Raylight integration for DiT tiling.

Provides AdvancedTilingRay node that works with raylight's RAY_ACTORS type,
applying toroidal attention tiling on distributed workers.

Uses manual Ray remote calls instead of @ray_patch to avoid serialization
issues with custom node packages that have non-standard import paths
(e.g. hyphens in directory names that prevent normal Python imports).
"""

try:
    from raylight.comfy_extra_dist.ray_patch_decorator import ray_patch
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
            gpu_workers = ray_actors["workers"]
            futures = [
                actor.model_function_runner.remote(
                    _remote_tiling_patch, settings.mode, settings.rotation
                )
                for actor in gpu_workers
            ]
            ray.get(futures)
            return (ray_actors,)


    def _remote_tiling_patch(model, mode, rotation):
        """Standalone tiling patch function for Ray workers.

        Loads the custom node package via importlib to avoid serialization
        issues with non-standard module paths (e.g. hyphens in directory names).

        Only primitive types (str, float) are passed as arguments so cloudpickle
        never needs to resolve the custom node module during deserialization.
        """
        import importlib.util
        import os
        import sys

        _node_dir = os.path.dirname(os.path.abspath(__file__))
        _pkg = "ComfyUI_AdvancedTiling"

        if _pkg not in sys.modules:
            _init = os.path.join(_node_dir, "__init__.py")
            spec = importlib.util.spec_from_file_location(
                _pkg, _init, submodule_search_locations=[_node_dir],
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[_pkg] = mod
            spec.loader.exec_module(mod)

        from ComfyUI_AdvancedTiling.dit_tiling import patch_dit_model
        from ComfyUI_AdvancedTiling.modes import Settings

        patch_dit_model(model, Settings(mode, rotation))
        return model
