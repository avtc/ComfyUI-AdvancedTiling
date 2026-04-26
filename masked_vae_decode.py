"""
Masked VAE decode for inpainting with minimal bleed.

At each stage of the VAE decoder, features in the non-masked (preserved)
area are replaced with features from a parallel decode of the source latent.
This prevents VAE convolution bleed from accumulating through the decoder
layers, reducing artifacts from ~16-24 pixels to ~1-3 pixels at mask
boundaries.

How it works:
  1. Decode the source latent, saving intermediate features at each stage
  2. Decode the inpainted latent, compositing at each stage:
     - masked area (1): keep inpainted features
     - preserved area (0): inject source features (resets accumulated bleed)
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


def _prepare_mask(mask, H_lat, W_lat, device):
    """
    Prepare mask at latent resolution.

    :param mask: Input mask (H, W), (1, H, W), or (B, H, W)
    :return: (1, 1, H_lat, W_lat) on device
    """
    m = mask.to(device=device, dtype=torch.float32)

    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif m.dim() == 3:
        m = m[:1].unsqueeze(1)  # (1, 1, H, W)

    if m.shape[2] != H_lat or m.shape[3] != W_lat:
        m = F.interpolate(
            m, size=(H_lat, W_lat), mode="bilinear", align_corners=False
        )

    return m


class InpaintVAEDecode:
    """
    VAE decode with intermediate compositing to eliminate bleed at mask
    boundaries.

    Takes an inpainted latent (after KSampler) and the source latent
    (before KSampler), along with the inpaint mask. Decodes both through
    the VAE decoder but, at each upsampling stage, replaces features in
    preserved areas with features from the source decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": (
                    "LATENT",
                    {"tooltip": "Inpainted latent (after KSampler)."},
                ),
                "source_samples": (
                    "LATENT",
                    {
                        "tooltip": "Source latent before sampling (from HexInpaint node). "
                        "Provides reference features for preserved areas."
                    },
                ),
                "vae": ("VAE", {"tooltip": "VAE model for decoding."}),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "Inpaint mask. "
                        "1 = inpainted area (keep from samples), "
                        "0 = preserved area (keep from source_samples)."
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

    def decode(self, samples, source_samples, vae, mask):
        z_inpaint = samples["samples"]
        z_source = source_samples["samples"]

        z_inpaint_4d, _ = _normalize_latent(z_inpaint)
        z_source_4d, _ = _normalize_latent(z_source)

        _, _, H_lat, W_lat = z_inpaint_4d.shape

        # Prepare mask at latent resolution on VAE device
        mask_lat = _prepare_mask(mask, H_lat, W_lat, vae.device)

        # Get compositing stages from the decoder
        decoder = vae.first_stage_model.decoder
        stages = _get_stage_modules(decoder)

        stage_names = [name for name, _ in stages]
        logger.info(
            f"[InpaintVAEDecode] Latent ({H_lat}x{W_lat}), "
            f"compositing stages: {stage_names}"
        )

        # Step 1: Reference decode — save features at each stage boundary
        ref_features = {}
        save_hooks = []
        for name, module in stages:
            save_hooks.append(
                module.register_forward_hook(_save_hook(ref_features, name))
            )

        with torch.no_grad():
            vae.decode(z_source)

        for h in save_hooks:
            h.remove()

        ref_summary = {k: tuple(v.shape) for k, v in ref_features.items()}
        logger.info(f"[InpaintVAEDecode] Reference features: {ref_summary}")

        # Step 2: Main decode — composite at each stage boundary
        comp_hooks = []
        for name, module in stages:
            comp_hooks.append(
                module.register_forward_hook(
                    _composite_hook(ref_features, mask_lat, name)
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

        logger.info(f"[InpaintVAEDecode] Output: {image.shape}")

        return (image,)
