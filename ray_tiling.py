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

        INPUT_IS_LIST=True: latent is a list (one per GPU worker).
        OUTPUT_IS_LIST=(False, True): actors single, settings list.
        """

        INPUT_IS_LIST = True
        OUTPUT_IS_LIST = (False, True)

        # pylint: disable=invalid-name

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "settings": ("ADVANCED_TILING_SETTINGS",),
                    "ray_actors": ("RAY_ACTORS",),
                    "vae": ("VAE",),
                    "latent": ("LATENT",),
                },
            }

        RETURN_TYPES = ("RAY_ACTORS", "RESOLVED_TILING_SETTINGS")
        RETURN_NAMES = ("ray_actors", "resolved_settings")
        FUNCTION = "run"
        CATEGORY = "conditioning"

        def run(self, settings_list, ray_actors_list, vae_list, latent_list):
            settings = settings_list[0]
            ray_actors = ray_actors_list[0]
            vae = vae_list[0]
            vae_factor = vae.spacial_compression_decode()

            # Resolve per-latent with image dimensions
            resolved_list = []
            for latent in latent_list:
                _, _, H_lat, W_lat = latent["samples"].shape
                img_W = W_lat * vae_factor
                img_H = H_lat * vae_factor
                # Raylight only supports DiT models (is_conv2d=False, patch_size=2)
                resolved = settings._resolve_auto(False, vae_factor, 2, img_W, img_H)
                resolved_list.append(resolved)

            if resolved_list[0].mode == "None":
                return (ray_actors, resolved_list)

            gpu_workers = ray_actors["workers"]

            # Defined inside run() so cloudpickle serializes it as a nested
            # function (by value) instead of by module reference.  Module-level
            # functions get serialized by reference, which requires importing
            # the module by name on the worker -- but the module name is the
            # filesystem path (with a hyphen), causing ModuleNotFoundError.
            #
            # Worker resolves settings with correct patch_size from the model
            # and returns the resolved settings to the master.
            def _patch(model, vae_factor, mode, rotation, scale, min_margin, divisible_by,
                       img_W, img_H):
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

                diff_model = model.model.diffusion_model
                patch_size = getattr(diff_model, 'patch_size', 1)
                raw = Settings(mode, rotation, scale, min_margin, divisible_by)
                resolved = raw._resolve_auto(False, vae_factor, patch_size, img_W, img_H)
                patch_dit_model(model, resolved)
                return resolved

            # Build per-worker args: each worker gets its latent's img dimensions
            worker_args = []
            for latent in latent_list:
                _, _, H_lat, W_lat = latent["samples"].shape
                worker_args.append((W_lat * vae_factor, H_lat * vae_factor))

            futures = [
                actor.model_function_runner.remote(
                    _patch, vae_factor, settings.mode, settings.rotation,
                    settings.scale, settings.min_margin, settings.divisible_by,
                    img_W, img_H,
                )
                for actor, (img_W, img_H) in zip(gpu_workers, worker_args)
            ]
            results = ray.get(futures)
            # Use resolved settings from workers (they have correct patch_size)
            resolved_list = results
            return (ray_actors, resolved_list)
