"""
Masked VAE decode for inpainting with minimal bleed.

At each stage of the VAE decoder, features in the non-masked (preserved)
area are replaced with features from a parallel decode of the reference
latent. This prevents VAE convolution bleed from accumulating through the
decoder layers, reducing artifacts from ~16-24 pixels to ~1-3 pixels at
mask boundaries.

When original_image is provided, it is VAE-encoded and used as the
reference latent instead of source_samples. This avoids bleed from
latent-space compositing boundaries (e.g. center+neighbor hex tiles)
that would otherwise contaminate the preserved area features.

How it works:
  1. Decode the reference latent, saving intermediate features at each stage
  2. Decode the inpainted latent, compositing at each stage:
     - masked area (1): keep inpainted features
     - preserved area (0): inject reference features (resets accumulated bleed)
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("ComfyUI-AdvancedTiling")


def _get_stage_modules(decoder):
    """
    Identify modules whose outputs represent decoder stage boundaries.

    Supports both standard Decoder (SD 1.5, SDXL, etc.) and Decoder3d
    (Wan, Qwen-Image, etc.) architectures.

    :return: List of (name, module) pairs
    """
    if hasattr(decoder, "conv_in"):
        return _stages_standard(decoder)
    elif hasattr(decoder, "conv1"):
        return _stages_3d(decoder)
    else:
        logger.warning(
            f"[InpaintVAEDecode] Unknown decoder type {type(decoder).__name__}, "
            f"no compositing stages found"
        )
        return []


def _stages_standard(decoder):
    """Stage detection for standard Decoder (SD 1.5, SDXL, etc.)."""
    stages = []

    # Stage 0: after initial convolution (latent resolution)
    stages.append(("conv_in", decoder.conv_in))

    # Stage 1: after mid block (latent resolution)
    stages.append(("mid", decoder.mid.block_2))

    # Stages 2+: after each up level
    num_res = decoder.num_resolutions
    for i_level in reversed(range(num_res)):
        up = decoder.up[i_level]
        if hasattr(up, "upsample"):
            stages.append((f"up_{i_level}", up.upsample))
        else:
            stages.append((f"up_{i_level}", up.block[-1]))

    return stages


def _stages_3d(decoder):
    """Stage detection for Decoder3d (Wan, Qwen-Image, etc.)."""
    stages = []

    # Stage 0: after initial convolution
    stages.append(("conv_in", decoder.conv1))

    # Stage 1: after middle blocks
    middle_layers = list(decoder.middle)
    if middle_layers:
        stages.append(("mid", middle_layers[-1]))

    # Stages 2+: after each spatial upsample in the sequential upsample blocks
    up_idx = 0
    for layer in decoder.upsamples:
        if type(layer).__name__ == "Resample":
            stages.append((f"up_{up_idx}", layer))
            up_idx += 1

    # If no Resample found, hook into last layer
    if up_idx == 0:
        ups = list(decoder.upsamples)
        if ups:
            stages.append(("up_last", ups[-1]))

    return stages


def _save_hook(features_dict, name):
    """Forward hook that saves module output to CPU."""

    def hook(module, input, output):
        features_dict[name] = output.detach().cpu()

    return hook


def _composite_hook(ref_features, mask_base, name):
    """Forward hook that composites output with reference features."""

    def hook(module, input, output):
        if name not in ref_features:
            return output

        ref = ref_features[name].to(device=output.device, dtype=output.dtype)

        if output.shape != ref.shape:
            return output

        # Spatial dimensions are always the last two
        H, W = output.shape[-2], output.shape[-1]

        # Scale mask to spatial resolution
        mask = mask_base
        if mask.shape[-2] != H or mask.shape[-1] != W:
            mask = F.interpolate(
                mask.float(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        # Expand mask dims to match output (handles 4D and 5D+ tensors)
        # mask starts as (1, 1, H, W), insert singleton dims for T etc.
        while mask.dim() < output.dim():
            mask = mask.unsqueeze(2)

        # masked area (1) → keep inpainted; preserved area (0) → use reference
        result = output * mask + ref * (1 - mask)
        return result.to(dtype=output.dtype)

    return hook


def _normalize_latent(latent):
    """Normalize latent to 4D (B, C, H, W) by squeezing singleton dims."""
    original_shape = latent.shape
    x = latent
    while x.dim() > 4:
        squeezed = False
        for d in range(1, x.dim() - 2):
            if x.shape[d] == 1:
                x = x.squeeze(d)
                squeezed = True
                break
        if not squeezed:
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
            break
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x, original_shape


def _prepare_mask(mask, device):
    """
    Prepare mask keeping original resolution.

    The compositing hook will downscale to each decoder stage's resolution,
    which preserves sharp boundaries better than downscaling to latent
    resolution first and then upscaling.

    :param mask: Input mask (H, W), (1, H, W), or (B, H, W)
    :return: (1, 1, H, W) on device
    """
    m = mask.to(device=device, dtype=torch.float32)

    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif m.dim() == 3:
        m = m[:1].unsqueeze(1)  # (1, 1, H, W)

    return m


class InpaintVAEDecode:
    """
    VAE decode with intermediate compositing to eliminate bleed at mask
    boundaries.

    Takes an inpainted latent (after KSampler) and a reference latent, along
    with the inpaint mask. Decodes both through the VAE decoder but, at each
    upsampling stage, replaces features in preserved areas with features from
    the reference decode.

    When original_image is provided, it is VAE-encoded and used as the
    reference instead of source_samples. This avoids bleed from latent-space
    compositing boundaries (e.g. center+neighbor hex tiles) that would
    otherwise contaminate the preserved-area features.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": (
                    "LATENT",
                    {"tooltip": "Inpainted latent (after KSampler)."},
                ),
                "vae": ("VAE", {"tooltip": "VAE model for decoding."}),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "Inpaint mask. "
                        "1 = inpainted area (keep from samples), "
                        "0 = preserved area (keep from reference)."
                    },
                ),
                "start_stage": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 6,
                        "tooltip": "Start compositing from this decoder stage. "
                        "Stage 0-1 are at latent resolution (coarse mask, may bleed "
                        "into border). Higher values skip early low-res stages for "
                        "cleaner boundaries. Stages: 0=conv_in, 1=mid, 2+=upsample.",
                    },
                ),
                "end_stage": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 6,
                        "tooltip": "Stop compositing after this decoder stage. "
                        "-1 = all remaining stages. Can be used to skip the final "
                        "high-res stage if it introduces artifacts.",
                    },
                ),
            },
            "optional": {
                "source_samples": (
                    "LATENT",
                    {
                        "tooltip": "Source latent before sampling (from HexInpaint node). "
                        "Used as reference when original_image is not provided."
                    },
                ),
                "original_image": (
                    "IMAGE",
                    {
                        "tooltip": "Original clean image. When provided, VAE-encoded and used "
                        "as the reference decode instead of source_samples. Avoids VAE "
                        "convolution bleed from latent-space compositing boundaries."
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent"
    DESCRIPTION = (
        "Decode inpainted latent with intermediate compositing to minimize "
        "VAE bleed at mask boundaries."
    )

    def decode(self, samples, vae, mask, start_stage=0, end_stage=-1,
               source_samples=None, original_image=None):
        z_inpaint = samples["samples"]

        if original_image is not None:
            z_ref = vae.encode(original_image)
        elif source_samples is not None:
            z_ref = source_samples["samples"]
        else:
            raise ValueError(
                "InpaintVAEDecode requires either source_samples or original_image "
                "to provide reference features for the preserved area."
            )

        z_inpaint_4d, _ = _normalize_latent(z_inpaint)
        z_ref_4d, _ = _normalize_latent(z_ref)

        _, _, H_lat, W_lat = z_inpaint_4d.shape

        # Prepare mask — keep at original resolution for sharp boundaries
        mask_prepared = _prepare_mask(mask, vae.device)

        # Get compositing stages from the decoder
        decoder = vae.first_stage_model.decoder
        stages = _get_stage_modules(decoder)

        # Determine which stages to composite
        last = len(stages) if end_stage < 0 else min(end_stage + 1, len(stages))
        composite_stages = set(range(start_stage, last))
        stage_names = [name for name, _ in stages]
        active_names = [stage_names[i] for i in sorted(composite_stages) if i < len(stages)]
        logger.info(
            f"[InpaintVAEDecode] Latent ({H_lat}x{W_lat}), "
            f"all stages: {stage_names}, compositing: {active_names}"
        )

        # Step 1: Reference decode — save features at stages we'll composite
        ref_features = {}
        save_hooks = []
        for i, (name, module) in enumerate(stages):
            if i in composite_stages:
                save_hooks.append(
                    module.register_forward_hook(_save_hook(ref_features, name))
                )

        with torch.no_grad():
            vae.decode(z_ref)

        for h in save_hooks:
            h.remove()

        ref_summary = {k: tuple(v.shape) for k, v in ref_features.items()}
        logger.info(f"[InpaintVAEDecode] Reference features: {ref_summary}")

        # Step 2: Main decode — composite only at selected stages
        comp_hooks = []
        for i, (name, module) in enumerate(stages):
            if i in composite_stages:
                comp_hooks.append(
                    module.register_forward_hook(
                        _composite_hook(ref_features, mask_prepared, name)
                    )
                )

        with torch.no_grad():
            image = vae.decode(z_inpaint)

        for h in comp_hooks:
            h.remove()

        # Handle 5D output (video VAEs)
        if image.ndim == 5:
            image = image.reshape(
                -1, image.shape[-3], image.shape[-2], image.shape[-1]
            )

        return (image,)
