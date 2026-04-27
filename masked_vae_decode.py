"""
Masked VAE decode for inpainting with minimal bleed.

At each stage of the VAE decoder, features in the non-masked (preserved)
area are replaced with features from a parallel decode of the reference
latent. This prevents VAE convolution bleed from accumulating through the
decoder layers, reducing artifacts from ~16-24 pixels to ~1-3 pixels at
mask boundaries.

The original_image is VAE-encoded and used as the
reference latent. This avoids bleed from latent-space compositing
boundaries (e.g. center+neighbor hex tiles) that would otherwise
contaminate the preserved area features.

How it works:
  1. Decode the reference latent, saving intermediate features at each stage
  2. Decode the inpainted latent, compositing at each stage:
     - masked area (1): keep inpainted features
     - preserved area (0): inject reference features (resets accumulated bleed)
"""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

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


def _compute_boundary_blend_map_from_mask(
    inside_mask: torch.Tensor,
    blend_pixels: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute blend map from an explicit inside mask tensor.

    :param inside_mask: Bool tensor (H, W) where True = inside hex.
    :param blend_pixels: Blend band width in pixels.
    :return: Tuple of (weights, nearest_y, nearest_x) tensors each (H, W).
    """
    H, W = inside_mask.shape
    inside_np = inside_mask.cpu().numpy().astype(np.float64)
    distances, indices = distance_transform_edt(inside_np, return_indices=True)

    nearest_y = torch.from_numpy(indices[0].astype(np.int64))
    nearest_x = torch.from_numpy(indices[1].astype(np.int64))

    band = (distances >= 1) & (distances <= blend_pixels)
    blend_values = np.zeros((H, W), dtype=np.float32)
    blend_values[band] = (blend_pixels - distances[band] + 1) / blend_pixels
    weights = torch.from_numpy(blend_values)

    return weights, nearest_y, nearest_x


def edge_extend_from_waste(
    latent: torch.Tensor,
    inside_mask: torch.Tensor,
    blend_pixels: int = 2,
) -> torch.Tensor:
    """Blend hex boundary pixels toward nearest waste-area pixel content.

    For each inside-hex pixel within ``blend_pixels`` of the hex boundary,
    blend its latent values toward the nearest outside-hex (waste) pixel.
    This ensures the cropped hex edge carries waste-area characteristics,
    producing seamless transitions when tiles are placed in a grid.

    Operates at latent resolution.  Through ~4 VAE upsampling stages the
    1-2 latent pixel blend propagates to 16-32 pixels at full resolution.

    :param latent: Latent tensor (B, C, H, W) — KSampler output.
    :param inside_mask: Bool tensor (H, W) where True = inside hex.
    :param blend_pixels: Blend band width in latent pixels (default 2).
    :return: Modified latent tensor with boundary blended toward waste content.
    """
    B, C, H, W = latent.shape

    if inside_mask.shape != (H, W):
        inside_mask = F.interpolate(
            inside_mask.float().unsqueeze(0).unsqueeze(0),
            size=(H, W), mode="nearest",
        ).squeeze(0).squeeze(0).bool()

    weights, nearest_y, nearest_x = _compute_boundary_blend_map_from_mask(
        inside_mask, blend_pixels,
    )

    weights = weights.to(latent.device)
    nearest_y = nearest_y.to(latent.device)
    nearest_x = nearest_x.to(latent.device)

    waste_content = latent[:, :, nearest_y, nearest_x]

    w = weights.unsqueeze(0).unsqueeze(0)
    return latent * (1 - w) + waste_content * w


def _feather_inner_edge(
    image: torch.Tensor,
    inside_mask: torch.Tensor,
    blend_band: int,
) -> torch.Tensor:
    """Blend inner hex edge toward nearest waste-area pixel content.

    Only affects a narrow band of pixels inside the hex boundary.
    Waste area and deep interior are untouched.

    :param image: (B, H, W, C) decoded image.
    :param inside_mask: (H, W) bool where True = inside hex.
    :param blend_band: Blend band width in pixels.
    :return: Modified image with feathered inner edge.
    """
    weights, nearest_y, nearest_x = _compute_boundary_blend_map_from_mask(
        inside_mask, blend_band,
    )

    weights = weights.to(image.device)
    nearest_y = nearest_y.to(image.device)
    nearest_x = nearest_x.to(image.device)

    waste_content = image[:, nearest_y, nearest_x, :]  # (B, H, W, C)

    w = weights.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
    return image * (1 - w) + waste_content * w


# ---------------------------------------------------------------------------
# Laplacian pyramid blending
# ---------------------------------------------------------------------------

def _gaussian_pyramid(img: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Build a Gaussian pyramid by successive 2x average-pooling.

    :param img: Tensor in (B, C, H, W) layout.
    :param levels: Number of down-sampling steps.
    :return: List of tensors from finest (original) to coarsest.
    """
    gp = [img]
    cur = img
    for _ in range(levels):
        cur = F.avg_pool2d(cur, kernel_size=2, stride=2)
        gp.append(cur)
    return gp


def _laplacian_pyramid(gp: list[torch.Tensor]) -> list[torch.Tensor]:
    """Build a Laplacian pyramid from a Gaussian pyramid.

    L[i] = G[i] - upsample(G[i+1])

    :param gp: Gaussian pyramid (finest to coarsest).
    :return: Laplacian pyramid (finest to coarsest).
    """
    lp = []
    for i in range(len(gp) - 1):
        H, W = gp[i].shape[-2], gp[i].shape[-1]
        up = F.interpolate(gp[i + 1], size=(H, W), mode="bilinear", align_corners=False)
        lp.append(gp[i] - up)
    lp.append(gp[-1])  # coarsest residue
    return lp


def laplacian_pyramid_blend(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    mask: torch.Tensor,
    levels: int = 0,
) -> torch.Tensor:
    """Blend two images using Laplacian pyramid with a smooth mask.

    :param image_a: (B, H, W, C) decoded inpainted image.
    :param image_b: (B, H, W, C) original_image reference.
    :param mask: (1, 1, H, W) smooth mask. 1.0 = keep *image_a*, 0.0 = keep *image_b*.
    :param levels: Pyramid depth. 0 = auto (``log2(min(H,W)) - 2``).
    :return: (B, H, W, C) blended image.
    """
    B, H, W, C = image_a.shape

    if levels <= 0:
        levels = max(1, int(math.log2(min(H, W))) - 2)

    # Convert to (B, C, H, W) float
    a = image_a.float().permute(0, 3, 1, 2)
    b = image_b.float().permute(0, 3, 1, 2)

    # Expand mask batch dimension to match images, ensure same device
    m = mask.float().to(device=a.device).expand(B, -1, -1, -1)  # (B, 1, H, W)

    # Build Gaussian pyramids
    gp_a = _gaussian_pyramid(a, levels)
    gp_b = _gaussian_pyramid(b, levels)
    gp_m = _gaussian_pyramid(m, levels)

    # Build Laplacian pyramids
    lp_a = _laplacian_pyramid(gp_a)
    lp_b = _laplacian_pyramid(gp_b)

    # Blend at each level
    blended_lp = []
    for i in range(len(lp_a)):
        g_mask = gp_m[i]
        blended = lp_a[i] * g_mask + lp_b[i] * (1 - g_mask)
        blended_lp.append(blended)

    # Reconstruct from coarsest to finest
    result = blended_lp[-1]
    for i in range(len(blended_lp) - 2, -1, -1):
        H_i, W_i = blended_lp[i].shape[-2], blended_lp[i].shape[-1]
        result = F.interpolate(result, size=(H_i, W_i), mode="bilinear", align_corners=False)
        result = result + blended_lp[i]

    # Clamp, convert back to (B, H, W, C)
    result = result.clamp(0, 1).permute(0, 2, 3, 1)
    return result.to(dtype=image_a.dtype)


def _make_smooth_hex_mask(
    waste_mask: torch.Tensor,
    blend_band: int,
    image_shape: tuple,
) -> torch.Tensor:
    """Create smooth hex mask for Laplacian pyramid blending.

    Inverts waste_mask (inside=1, waste=0) and applies Gaussian blur
    to create a smooth transition zone at the hex boundary.

    :param waste_mask: (1, H, W) or (H, W), waste=1, inside=0.
    :param blend_band: Transition band width in pixels.
    :param image_shape: (B, H, W, C) of the target image.
    :return: (1, 1, H, W) float tensor, 1.0=inside hex, 0.0=waste.
    """
    # 1. Invert: inside becomes 1, waste becomes 0
    mask = 1.0 - waste_mask.float()

    # 2. Reshape to (1, 1, H, W)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif mask.dim() == 3:
        mask = mask[:1].unsqueeze(1)  # (1, 1, H, W)

    # 3. Resize to match image_shape spatial dims if needed
    _, _, H, W = mask.shape
    target_H, target_W = image_shape[1], image_shape[2]
    if H != target_H or W != target_W:
        mask = F.interpolate(
            mask, size=(target_H, target_W), mode="bilinear", align_corners=False,
        )

    # 4. Gaussian blur
    sigma = blend_band / 3.0
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1  # ensure odd

    # Create 1D Gaussian kernel
    x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Create 2D kernel via outer product, shape (1, 1, k, k)
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

    # Apply with reflect padding
    pad = ksize // 2
    mask = F.pad(mask, [pad, pad, pad, pad], mode="reflect")
    mask = F.conv2d(mask, kernel_2d)

    return mask.clamp(0.0, 1.0)


class InpaintVAEDecode:
    """
    VAE decode with intermediate compositing to eliminate bleed at mask
    boundaries.

    Takes an inpainted latent (after KSampler) and an original image, along
    with the inpaint mask. Decodes both through the VAE decoder but, at each
    upsampling stage, replaces features in preserved areas with features from
    the reference decode.

    The original_image is VAE-encoded and used as the reference decode.
    This avoids bleed from latent-space compositing boundaries
    (e.g. center+neighbor hex tiles) that would otherwise contaminate the
    preserved-area features.
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
                "waste_mask": (
                    "MASK",
                    {
                        "tooltip": "Waste-area mask from HexInpaint. "
                        "waste=1 (keep from samples), "
                        "inside=0 (inject from reference). "
                        "Also defines hex geometry for edge-extend."
                    },
                ),
                "original_image": (
                    "IMAGE",
                    {
                        "tooltip": "Original clean image. VAE-encoded and used "
                        "as the reference decode. Avoids VAE convolution bleed "
                        "from latent-space compositing boundaries."
                    },
                ),
                "stages": ("STRING", {
                    "default": "all",
                    "tooltip": "Comma-separated list of decoder stages to composite. "
                    "'all' = all stages. Example: '1,4' for stages 1 and 4 only. "
                    "Stages: 0=conv_in, 1=mid, 2+=upsample (varies by VAE).",
                }),
                "edge_extend": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Blend hex boundary pixels toward waste area before decode. Produces seamless transitions when cropped tiles are placed in a grid.",
                }),
                "inject_waste": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Inject original neighbor content into waste area at each VAE upscale stage. "
                    "Disable for single-pass decode with edge-extend only.",
                }),
                "laplacian_blend": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Post-process with Laplacian pyramid blending at hex boundary. "
                    "Multi-scale blend between decoded and original_image.",
                }),
                "feather_restore": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Restore waste area from original_image with feathered transition at hex boundary. "
                    "Waste area is replaced pixel-perfect; only a narrow band at the hex edge is blended.",
                }),
                "blend_band": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Transition band width in pixels at hex boundary.",
                }),
            },
            "optional": {
                "waste_mask_img": (
                    "MASK",
                    {
                        "tooltip": "Waste-area mask at image resolution from HexInpaint. "
                        "When provided, used for pixel-perfect Laplacian pyramid blending. "
                        "Falls back to upscaling waste_mask if not connected.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent"
    DESCRIPTION = (
        "Decode inpainted latent with intermediate compositing to minimize "
        "VAE bleed at mask boundaries."
    )

    def decode(self, samples, vae, waste_mask, original_image, stages="all",
               edge_extend=False, inject_waste=True,
               laplacian_blend=False, feather_restore=False, blend_band=8,
               waste_mask_img=None):
        z_inpaint = samples["samples"].clone()

        # Edge-extend: blend hex boundary toward waste area
        if edge_extend:
            inside = (waste_mask == 0)
            if inside.dim() == 3:
                inside = inside[0]
            z_4d, orig_shape = _normalize_latent(z_inpaint)
            z_4d = edge_extend_from_waste(z_4d, inside)
            z_inpaint = z_4d.reshape(orig_shape)

        # Decode
        if inject_waste:
            # Two-pass decode with per-stage compositing
            z_ref = vae.encode(original_image)

            z_inpaint_4d, _ = _normalize_latent(z_inpaint)
            z_ref_4d, _ = _normalize_latent(z_ref)

            _, _, H_lat, W_lat = z_inpaint_4d.shape

            # Prepare compositing mask: inject reference into waste area (waste=1)
            # Invert: inside=1 (keep output), waste=0 (inject reference)
            # Use image-res mask when available for sharper boundaries at high-res stages
            compositing_source = waste_mask_img if waste_mask_img is not None else waste_mask
            compositing_mask = 1.0 - compositing_source.float()
            mask_prepared = _prepare_mask(compositing_mask, vae.device)

            # Get compositing stages from the decoder
            decoder = vae.first_stage_model.decoder
            decoder_stages = _get_stage_modules(decoder)

            # Determine which stages to composite
            num_stages = len(decoder_stages)
            if stages.strip().lower() == "all":
                composite_stages = set(range(num_stages))
            else:
                composite_stages = set()
                for part in stages.split(","):
                    s = part.strip()
                    if s:
                        idx = int(s)
                        if 0 <= idx < num_stages:
                            composite_stages.add(idx)
            stage_names = [name for name, _ in decoder_stages]
            active_names = [stage_names[i] for i in sorted(composite_stages) if i < len(decoder_stages)]
            logger.info(
                f"[InpaintVAEDecode] Latent ({H_lat}x{W_lat}), "
                f"all stages: {stage_names}, compositing: {active_names}"
            )

            # Step 1: Reference decode — save features at stages we'll composite
            ref_features = {}
            save_hooks = []
            for i, (name, module) in enumerate(decoder_stages):
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
            for i, (name, module) in enumerate(decoder_stages):
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
        else:
            # Single-pass decode
            with torch.no_grad():
                image = vae.decode(z_inpaint)
            if image.ndim == 5:
                image = image.reshape(
                    -1, image.shape[-3], image.shape[-2], image.shape[-1]
                )

        # Post-processing: smooth seam at hex boundary
        if feather_restore or laplacian_blend:
            if feather_restore:
                # Blend inner hex edge toward nearest waste-area pixel content.
                # Only affects a narrow band inside the hex boundary.
                # Waste area and deep interior are untouched.
                blend_source = waste_mask_img if waste_mask_img is not None else waste_mask
                inside = (blend_source == 0)
                if inside.dim() == 3:
                    inside = inside[0]
                H_img, W_img = image.shape[1], image.shape[2]
                if inside.shape != (H_img, W_img):
                    inside = F.interpolate(
                        inside.float().unsqueeze(0).unsqueeze(0),
                        size=(H_img, W_img), mode="nearest",
                    ).squeeze(0).squeeze(0).bool()
                image = _feather_inner_edge(image, inside, blend_band)
            else:
                ref = original_image.to(device=image.device, dtype=image.dtype)
                blend_mask = waste_mask_img if waste_mask_img is not None else waste_mask
                mask = _make_smooth_hex_mask(blend_mask, blend_band, image.shape)
                image = laplacian_pyramid_blend(image, ref, mask)

        return (image,)
