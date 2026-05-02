"""
Hex tile inpainting: composites center + neighbor latents with border masks
for natural edge transitions.

Uses hex geometry directly (not through settings tiling mode) so that:
- Masks and compositing always use hex_tiling
- No Conv2d wrapping is applied (preserves neighbor content in waste region)
"""

import time
import logging
import functools

import torch

from .modes import Settings
from .modes.hex_mask import (
    NEIGHBOR_DIRECTIONS,
    compute_edge_buffer_mask,
    create_corner_masks,
    create_feathered_masks,
    create_waste_mask,
    _neighbor_offset_px,
)
from .corner_composite import (
    CORNER_NAMES,
    CORNER_NEIGHBORS,
    build_corner_tile_map,
    composite_corner_latents,
    composite_corner_preview,
    get_corner_offsets,
)
from .tile_priority import match_terrain_priority

logger = logging.getLogger("ComfyUI-AdvancedTiling")

# Per-direction color for mask visualization and regional prompting
DIRECTION_COLORS = torch.tensor([
    [1.0, 0.0, 0.0],  # E  → Red
    [0.0, 0.0, 1.0],  # NE → Blue
    [0.0, 1.0, 0.0],  # NW → Green
    [1.0, 1.0, 0.0],  # W  → Yellow
    [1.0, 0.5, 0.0],  # SW → Orange
    [0.5, 0.0, 1.0],  # SE → Purple
])
DIRECTION_COLOR_NAMES = ["red", "blue", "green", "yellow", "orange", "purple"]


def _normalize_latent(latent: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """
    Normalize latent to 4D (B, C, H, W) by squeezing singleton temporal dims.

    Video/3D VAEs (e.g. wan) return 5D tensors (1, C, T, H, W).
    This squeezes singleton dims between channel and spatial to get 4D.

    :return: (4D tensor, original shape for restoration)
    """
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


def _paste_with_offset(dst, src, mask, direction, hex_radius, is_latent=True):
    """
    Paste src into dst at mask positions with hex-neighbor coordinate offset.

    Maps output (y,x) to src coordinates by subtracting the inter-hex offset
    for the given direction. This ensures correct spatial alignment between
    the center tile and neighbor tiles.

    :param dst: Destination tensor (modified in-place)
    :param src: Source tensor (same batch/chan layout as dst)
    :param mask: Boolean mask (H, W) of pixels to paste
    :param direction: Neighbor direction ("E", "NE", etc.)
    :param hex_radius: Hex circumradius in pixels
    :param is_latent: True for (B,C,H,W), False for (B,H,W,C)
    """
    H, W = mask.shape
    ox, oy = _neighbor_offset_px(direction, hex_radius)
    ox_i, oy_i = round(ox), round(oy)
    ys, xs = torch.where(mask)
    src_ys = (ys - oy_i).clamp(0, H - 1)
    src_xs = (xs - ox_i).clamp(0, W - 1)
    if is_latent:
        dst[:, :, ys, xs] = src[:, :, src_ys, src_xs]
    else:
        dst[0, ys, xs] = src[0, src_ys, src_xs]


@functools.cache
def _build_neighbor_map(
    width: int, height: int, rotation: float
) -> torch.Tensor:
    """
    Build a map from pixel position to neighbor direction index.

    Uses hex_tiling directly (not settings.tiling_fn) so the hex geometry
    is always correct regardless of the settings mode.

    For each pixel outside the hex (waste region), assigns a direction
    index (0-5 for E/NE/NW/W/SW/SE). Pixels inside the hex get -1.

    :return: LongTensor of shape (height, width) with values -1 to 5
    """
    from .modes.hex_vectorized import hex_tiling_vectorized
    from .modes.hex_mask import _compute_sector_map

    hex_size = min(width, height) // 2
    mapped_x, mapped_y = hex_tiling_vectorized(width, height, rotation, hex_size)

    mapped_x_t = torch.from_numpy(mapped_x)
    mapped_y_t = torch.from_numpy(mapped_y)

    xs = torch.arange(width).expand(height, width)
    ys = torch.arange(height).unsqueeze(1).expand(height, width)
    is_waste = (mapped_x_t != xs) | (mapped_y_t != ys)

    sectors = _compute_sector_map(width, height)

    neighbor_map = torch.full((height, width), -1, dtype=torch.long)
    neighbor_map[is_waste] = sectors[is_waste]
    return neighbor_map


def composite_latents(
    center_latent: torch.Tensor,
    neighbor_latents: dict[str, torch.Tensor],
    settings: Settings,
    rotation_steps: int = 0,
) -> torch.Tensor:
    """
    Composite center and neighbor latents into a single latent.

    Places neighbor content in the waste region around the center hex.
    Uses coordinate offsets to map output positions to neighbor latent
    positions (each neighbor hex is centered at a different position).

    :param center_latent: Center tile latent
    :param neighbor_latents: Dict mapping direction name ("E", "NE", etc.)
                             to neighbor latent
    :param settings: Tiling settings (rotation only)
    :param rotation_steps: Rotate direction mapping by N steps clockwise
    :return: Composited latent (same shape as input)
    """
    t0 = time.time()
    result, original_shape = _normalize_latent(center_latent)
    B, C, H, W = result.shape

    t1 = time.time()
    neighbor_map = _build_neighbor_map(W, H, settings.rotation)
    t2 = time.time()

    total_waste = (neighbor_map >= 0).sum().item()
    logger.info(f"composite_latents: waste={total_waste}/{H*W}, "
                f"shape=({B},{C},{H},{W}), "
                f"normalize={t1-t0:.3f}s, neighbor_map={t2-t1:.3f}s")

    n = len(NEIGHBOR_DIRECTIONS)
    direction_to_idx = {name: idx for idx, name in enumerate(NEIGHBOR_DIRECTIONS)}
    hex_radius = min(W, H) // 2

    total_pasted = 0
    for direction_name, neighbor_latent in neighbor_latents.items():
        dir_idx = direction_to_idx[direction_name]
        target_idx = (dir_idx - rotation_steps) % n
        mask = (neighbor_map == target_idx)  # [H, W] boolean
        count = mask.sum().item()

        if count == 0:
            logger.info(f"  {direction_name}: no waste pixels found")
            continue

        neighbor_4d, _ = _normalize_latent(neighbor_latent)
        _paste_with_offset(result, neighbor_4d, mask, direction_name, hex_radius, is_latent=True)
        total_pasted += count
        logger.info(f"  {direction_name}: pasted {count} pixels")

    logger.info(f"  total pasted: {total_pasted}/{total_waste} waste pixels")

    if len(original_shape) > 4:
        result = result.reshape(original_shape)
    return result


class AdvancedTilingHexInpaint:
    """
    Hex tile inpainting node. Composites center + neighbor latents,
    generates border masks for natural edge transitions.

    Uses hex geometry directly — does NOT apply Conv2d wrapping or depend
    on the settings tiling mode. The settings only provide hex rotation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "settings": ("ADVANCED_TILING_SETTINGS",),
                "vae": ("VAE",),
                "center_image": ("IMAGE",),
                "inpaint_mode": (
                    ["CentralTile", "CornerEdges"],
                    {
                        "default": "CentralTile",
                        "tooltip": "CentralTile: inpaint edges of central hex tile. CornerEdges: compose 3 tiles meeting at a corner, inpaint the 3 edges between them.",
                    },
                ),
                "border_width": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.05,
                        "max": 0.45,
                        "step": 0.01,
                        "tooltip": "Width of the inpainting border as fraction of hex radius.",
                    },
                ),
                "feather_radius": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Feather radius as fraction of border width. Softens inner and inactive-side edges for smoother transitions. 0 = sharp edges.",
                    },
                ),
                "feather_sides": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, also feathers the side edges of each mask where no active neighbor is present. Disable to feather only the inner edge (toward hex center).",
                    },
                ),
                "mask_strength_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Minimum denoise strength in the masked area. Non-zero mask values are remapped so the softest feather edge applies at least this much denoising.",
                    },
                ),
                "mask_strength_max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Maximum denoise strength in the masked area. Non-zero mask values are remapped so the strongest area caps at this value.",
                    },
                ),
                "skip_same_neighbors": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, automatically skips inpainting borders where the neighbor image is identical to the center image. Useful when surrounding tiles share the same terrain type.",
                    },
                ),
                "rotate_mapping": (
                    ["0", "1", "2", "3", "4", "5", "ALL"],
                    {
                        "default": "0",
                        "tooltip": "Rotate neighbor image assignments by N steps clockwise. ALL: output all 6 rotations as list.",
                    },
                ),
                "enable_preview": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable composited preview output (VAE decode of center + neighbor latents). Adds ~0.5-2s per run.",
                    },
                ),
                "corner_select": (
                    ["N", "NE", "SE", "S", "SW", "NW", "ALL"],
                    {
                        "default": "NE",
                        "tooltip": "Which corner to process in CornerEdges mode. Determines which 2 neighbors are used.",
                    },
                ),
                "mask_extent": (
                    ["half_edge", "full_edge"],
                    {
                        "default": "half_edge",
                        "tooltip": "half_edge: mask extends to edge midpoint (composable corners). full_edge: mask covers entire edge.",
                    },
                ),
            },
            "optional": {
                **{
                    f"neighbor_{d}": ("IMAGE", {"tooltip": f"{d} neighbor tile image"})
                    for d in NEIGHBOR_DIRECTIONS
                },
                "priorities": ("TERRAIN_PRIORITIES", {
                    "tooltip": "Terrain priority settings. Determines which edges get inpainted based on terrain type.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "IMAGE", "IMAGE", "LATENT", "IMAGE", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = (
        "LATENT", "MASK",
        *[f"{NEIGHBOR_DIRECTIONS[i]}_{DIRECTION_COLOR_NAMES[i]}"
          for i in range(len(NEIGHBOR_DIRECTIONS))],
        "composited_preview",
        "color_mask",
        "LATENT_paintbrush",
        "paintbrush_preview",
        "WASTE_MASK",
        "WASTE_MASK_IMG",
        "OVERLAP_IMAGE",
    )
    OUTPUT_IS_LIST = (True,) * 15
    FUNCTION = "run"
    CATEGORY = "conditioning"

    def run(self, settings, vae, center_image, inpaint_mode, border_width,
            feather_radius, feather_sides, mask_strength_min, mask_strength_max,
            skip_same_neighbors, rotate_mapping="0", enable_preview=False,
            corner_select="NE", mask_extent="half_edge",
            **kwargs):
        t_start = time.time()

        # 1. VAE-encode center image (once, shared across iterations)
        t0 = time.time()
        center_latent = vae.encode(center_image)
        logger.info(f"[HexInpaint] VAE encode center: {time.time()-t0:.3f}s, "
                     f"image={center_image.shape}, latent={center_latent.shape}")

        # 2. VAE-encode provided neighbor images (once, shared)
        neighbor_latents = {}
        neighbor_images = {}
        for direction in NEIGHBOR_DIRECTIONS:
            key = f"neighbor_{direction}"
            if key in kwargs and kwargs[key] is not None:
                t_n = time.time()
                neighbor_latents[direction] = vae.encode(kwargs[key])
                neighbor_images[direction] = kwargs[key]
                logger.info(f"[HexInpaint] VAE encode {direction}: "
                             f"{time.time()-t_n:.3f}s, latent={neighbor_latents[direction].shape}")

        logger.info(f"[HexInpaint] Neighbors provided: {list(neighbor_latents.keys())}")

        # 3. Determine iteration list
        if inpaint_mode == "CentralTile":
            if rotate_mapping == "ALL":
                iterations = [(inpaint_mode, r, "NE") for r in range(6)]
            else:
                iterations = [(inpaint_mode, int(rotate_mapping) % 6, "NE")]
        else:  # CornerEdges
            if corner_select == "ALL":
                iterations = [(inpaint_mode, 0, c) for c in CORNER_NAMES]
            else:
                iterations = [(inpaint_mode, 0, corner_select)]

        # 4. Run each iteration
        all_results = []
        for mode, rot, corner in iterations:
            result = self._run_single(
                settings, vae, center_image, center_latent, neighbor_latents,
                neighbor_images, border_width, feather_radius, feather_sides,
                mask_strength_min, mask_strength_max, skip_same_neighbors,
                enable_preview, mask_extent, kwargs,
                mode, rot, corner,
            )
            all_results.append(result)

        # 5. Transpose: list of 15-tuples -> 15 lists
        num_outputs = 15
        transposed = tuple(
            [result[i] for result in all_results]
            for i in range(num_outputs)
        )

        logger.info(f"[HexInpaint] Total: {time.time()-t_start:.3f}s, "
                     f"iterations={len(all_results)}")
        return transposed

    def _run_single(
        self, settings, vae, center_image, center_latent, neighbor_latents,
        neighbor_images, border_width, feather_radius, feather_sides,
        mask_strength_min, mask_strength_max, skip_same_neighbors,
        enable_preview, mask_extent, kwargs,
        inpaint_mode, rotation_steps, corner_select,
    ):
        """Run the pipeline for one rotation/corner. Returns tuple of 15 outputs (not lists)."""
        if inpaint_mode == "CornerEdges":
            return self._run_corner_edges(
                settings, vae, center_image, center_latent, neighbor_latents,
                neighbor_images, border_width, feather_radius, feather_sides,
                mask_strength_min, mask_strength_max, corner_select, mask_extent,
                enable_preview, kwargs,
            )

        # --- CentralTile mode ---
        t_start = time.time()
        n = len(NEIGHBOR_DIRECTIONS)

        if rotation_steps:
            logger.info(f"[HexInpaint] Mapping rotated by {rotation_steps} steps CW")

        # 3. Composite latents with rotated direction mapping
        if neighbor_latents:
            t2 = time.time()
            composited = composite_latents(
                center_latent, neighbor_latents, settings, rotation_steps
            )
            logger.info(f"[HexInpaint] Composite: {time.time()-t2:.3f}s")
        else:
            composited = center_latent
            logger.info("[HexInpaint] No neighbor images — skipping compositing")

        # 4. Determine active directions (have non-matching neighbor)
        active_directions = set()
        for i, direction in enumerate(NEIGHBOR_DIRECTIONS):
            if direction in neighbor_images:
                nimg = neighbor_images[direction]
                if nimg.shape != center_image.shape:
                    logger.warning(f"[HexInpaint] {direction} size mismatch "
                                   f"({nimg.shape} vs {center_image.shape}), treating as active")
                    active_directions.add((i - rotation_steps) % n)
                elif skip_same_neighbors and torch.allclose(center_image, nimg, atol=1e-6):
                    logger.info(f"[HexInpaint] Auto-skip {direction}: matches center")
                else:
                    active_directions.add((i - rotation_steps) % n)

        if active_directions:
            logger.info(f"[HexInpaint] Active: "
                         f"{[NEIGHBOR_DIRECTIONS[i] for i in sorted(active_directions)]}")
        else:
            logger.info("[HexInpaint] No active directions")

        # 4b. Determine per-direction priorities for mask filtering.
        # Only mask sectors where the neighbor has higher priority (lower number)
        # than the center — those borders need regeneration to blend with the
        # higher-priority neighbor. Sectors where center has higher or equal
        # priority are excluded from the mask (preserve center content).
        priorities = kwargs.get("priorities")
        center_pri = match_terrain_priority(center_image, priorities) if priorities else None
        masked_directions = active_directions  # default: all active directions
        if center_pri is not None and priorities is not None:
            masked_directions = set()
            for i, direction in enumerate(NEIGHBOR_DIRECTIONS):
                target_idx = (i - rotation_steps) % n
                if target_idx not in active_directions:
                    continue
                if direction in neighbor_images:
                    n_pri = match_terrain_priority(neighbor_images[direction], priorities)
                    if n_pri is not None and n_pri < center_pri:
                        masked_directions.add(target_idx)
                        logger.info(f"[HexInpaint] {direction}(pri={n_pri}) > center(pri={center_pri}): mask")
                    else:
                        logger.info(f"[HexInpaint] {direction}(pri={n_pri}) <= center(pri={center_pri}): skip")
            if not masked_directions:
                logger.info("[HexInpaint] No directions to mask (center has highest priority)")

        # 5. Generate masks at latent resolution.
        t3 = time.time()
        composited_4d, _ = _normalize_latent(composited)
        _, _, H_lat, W_lat = composited_4d.shape
        H_img, W_img = center_image.shape[1], center_image.shape[2]

        hex_radius_lat = min(W_lat, H_lat) // 2
        erosion_lat = max(1, int(border_width * hex_radius_lat))
        feather_lat = max(0, round(feather_radius * erosion_lat))

        _, border_mask_lat, neighbor_masks_lat = create_feathered_masks(
            W_lat, H_lat, settings, border_width, feather_lat, feather_sides,
            masked_directions,
        )

        # Waste-area mask at latent resolution for InpaintVAEDecode
        waste_mask_lat = create_waste_mask(W_lat, H_lat, settings)  # (1, H_lat, W_lat)
        waste_mask_img = create_waste_mask(W_img, H_img, settings)  # (1, H_img, W_img)

        # 5b. Edge buffer: paste neighbor latent into the outermost pixels of
        # the border ring per direction, then exclude those pixels from the
        # inpaint mask. This seeds the diffusion boundary with correct adjacent
        # content so VAE decode doesn't bleed inpainted changes into edges.
        # Uses edge_buffer_depth=0 (boundary only) matching the proven approach.
        if neighbor_latents and masked_directions:
            buffer_dir_masks = compute_edge_buffer_mask(
                W_lat, H_lat, settings.rotation, border_width, 0,
                masked_directions,
            )
            direction_to_idx = {name: idx for idx, name in enumerate(NEIGHBOR_DIRECTIONS)}
            total_buffer = 0
            for direction_name, neighbor_latent in neighbor_latents.items():
                dir_idx = direction_to_idx[direction_name]
                target_idx = (dir_idx - rotation_steps) % n
                if target_idx not in masked_directions:
                    continue
                buf_mask = buffer_dir_masks[target_idx]
                if buf_mask.any():
                    neighbor_4d, _ = _normalize_latent(neighbor_latent)
                    composited_4d[:, :, buf_mask] = neighbor_4d[:, :, buf_mask]
                    total_buffer += buf_mask.sum().item()
            # Exclude edge buffer from inpaint mask
            buffer_combined = buffer_dir_masks.any(dim=0)
            border_mask_lat = torch.where(
                buffer_combined.unsqueeze(0),
                torch.zeros_like(border_mask_lat),
                border_mask_lat,
            )
            if total_buffer:
                logger.info(f"[HexInpaint] Edge buffer: pixels={total_buffer}")

        t4 = time.time()
        logger.info(f"[HexInpaint] Latent masks ({W_lat}x{H_lat}): {t4-t3:.3f}s, "
                     f"border={int(border_mask_lat.sum().item())}, feather={feather_lat}px")

        # 6. Remap mask strength range (non-zero values only)
        if mask_strength_min > 0.0 or mask_strength_max < 1.0:
            nonzero = border_mask_lat > 0
            border_mask_lat = torch.where(
                nonzero,
                mask_strength_min + border_mask_lat * (mask_strength_max - mask_strength_min),
                border_mask_lat,
            )
            logger.info(f"[HexInpaint] Strength remap: "
                         f"[0,1] -> [{mask_strength_min:.2f},{mask_strength_max:.2f}]")

        # 7. Build latent dict with noise_mask
        noise_mask = border_mask_lat.unsqueeze(0)  # (1, 1, H_lat, W_lat)
        latent_dict = {"samples": composited, "noise_mask": noise_mask}
        logger.info(f"[HexInpaint] noise_mask shape={noise_mask.shape}, "
                     f"coverage={noise_mask.mean().item():.3f}")

        # 8. Generate masks at image resolution (for output visualization)
        t5 = time.time()
        hex_radius_img = min(W_img, H_img) // 2
        erosion_img = max(1, int(border_width * hex_radius_img))
        feather_img = max(0, round(feather_radius * erosion_img))

        _, full_border_mask, full_neighbor_masks = create_feathered_masks(
            W_img, H_img, settings, border_width, feather_img, feather_sides, active_directions
        )
        logger.info(f"[HexInpaint] Image masks ({W_img}x{H_img}): {time.time()-t5:.3f}s")

        # 9. Remap image-resolution masks to match
        if mask_strength_min > 0.0 or mask_strength_max < 1.0:
            nonzero_img = full_border_mask > 0
            full_border_mask = torch.where(
                nonzero_img,
                mask_strength_min + full_border_mask * (mask_strength_max - mask_strength_min),
                full_border_mask,
            )
            for i in range(len(NEIGHBOR_DIRECTIONS)):
                nz = full_neighbor_masks[i] > 0
                full_neighbor_masks[i] = torch.where(
                    nz,
                    mask_strength_min + full_neighbor_masks[i] * (mask_strength_max - mask_strength_min),
                    full_neighbor_masks[i],
                )

        # 10. Assemble outputs
        outputs = [latent_dict, full_border_mask]
        for i in range(len(NEIGHBOR_DIRECTIONS)):
            outputs.append(full_neighbor_masks[i])

        # Build neighbor map (shared by preview and overlap)
        neighbor_map_img = _build_neighbor_map(W_img, H_img, settings.rotation)
        dir_idx_map = {name: idx for idx, name in enumerate(NEIGHBOR_DIRECTIONS)}

        # Overlap image: center + neighbor content in waste area AND border mask.
        # Extends adjacent content into the hex boundary so InpaintVAEDecode
        # sees correct adjacent content at the inpainting edge.
        overlap_image = center_image.clone()
        for direction_name, neighbor_img in neighbor_images.items():
            dir_idx = dir_idx_map[direction_name]
            target_idx = (dir_idx - rotation_steps) % n
            waste_region = (neighbor_map_img == target_idx)
            if waste_region.any():
                _paste_with_offset(overlap_image, neighbor_img, waste_region,
                                   direction_name, hex_radius_img, is_latent=False)
            border_region = full_neighbor_masks[target_idx] > 0
            if border_region.any():
                _paste_with_offset(overlap_image, neighbor_img, border_region,
                                   direction_name, hex_radius_img, is_latent=False)

        # output 9: composited preview (tiles arrangement as passed to latent)
        if enable_preview:
            preview_image = center_image.clone()
            for direction_name, neighbor_img in neighbor_images.items():
                dir_idx = dir_idx_map[direction_name]
                target_idx = (dir_idx - rotation_steps) % n
                waste_region = (neighbor_map_img == target_idx)
                if waste_region.any():
                    _paste_with_offset(preview_image, neighbor_img, waste_region,
                                       direction_name, hex_radius_img, is_latent=False)
            outputs.append(preview_image)
        else:
            outputs.append(torch.zeros(1, 1, 1, 3, dtype=torch.float32))

        # output 10: color mask for regional prompting
        color_mask = torch.zeros(H_img, W_img, 3, dtype=torch.float32)
        for i in range(len(NEIGHBOR_DIRECTIONS)):
            mask_i = full_neighbor_masks[i]  # (H_img, W_img)
            color = DIRECTION_COLORS[i]
            for c in range(3):
                color_mask[:, :, c] += mask_i * color[c]
        outputs.append(color_mask.unsqueeze(0))  # (1, H, W, 3)

        # output 11: paintbrush latent (image with solid colors in border regions)
        # Build paintbrush image: original + solid color overlay in border
        solid_color = torch.zeros(H_img, W_img, 3, dtype=torch.float32)
        border_any = torch.zeros(H_img, W_img, dtype=torch.float32)
        for i in range(len(NEIGHBOR_DIRECTIONS)):
            region = (full_neighbor_masks[i] > 0).float()
            border_any = torch.max(border_any, region)
            color = DIRECTION_COLORS[i]
            for c in range(3):
                solid_color[:, :, c] = torch.where(
                    region > 0, color[c], solid_color[:, :, c],
                )
        border_3ch = border_any.unsqueeze(-1)
        paintbrush_img = center_image[0] * (1 - border_3ch) + solid_color * border_3ch
        paintbrush_img = paintbrush_img.unsqueeze(0)  # (1, H, W, 3)

        # Composite neighbor images into waste region of paintbrush preview
        for direction_name, neighbor_img in neighbor_images.items():
            dir_idx = dir_idx_map[direction_name]
            target_idx = (dir_idx - rotation_steps) % n
            waste_region = (neighbor_map_img == target_idx)
            if waste_region.any():
                _paste_with_offset(paintbrush_img, neighbor_img, waste_region,
                                   direction_name, hex_radius_img, is_latent=False)

        t_pb = time.time()
        paintbrush_latent = vae.encode(paintbrush_img)
        if neighbor_latents:
            paintbrush_composited = composite_latents(
                paintbrush_latent, neighbor_latents, settings, rotation_steps,
            )
        else:
            paintbrush_composited = paintbrush_latent
        paintbrush_latent_dict = {
            "samples": paintbrush_composited,
            "noise_mask": noise_mask,
        }
        outputs.append(paintbrush_latent_dict)
        outputs.append(paintbrush_img)
        logger.info(f"[HexInpaint] Paintbrush latent: {time.time()-t_pb:.3f}s")

        # output 13: waste-area mask (latent resolution)
        outputs.append(waste_mask_lat)
        # output 14: waste-area mask (image resolution)
        outputs.append(waste_mask_img)
        # output 15: overlap image (center + neighbors in waste area)
        outputs.append(overlap_image)

        logger.info(f"[HexInpaint] Total: {time.time()-t_start:.3f}s")
        return tuple(outputs)

    def _run_corner_edges(
        self, settings, vae, center_image, center_latent, neighbor_latents,
        neighbor_images, border_width, feather_radius, feather_sides,
        mask_strength_min, mask_strength_max, corner_select, mask_extent,
        enable_preview, kwargs,
    ):
        """CornerEdges mode: compose 3 tiles at a corner and generate edge masks."""
        from .tile_priority import TerrainPriorities

        t_start = time.time()
        priorities = kwargs.get("priorities")

        n1_dir, n2_dir = CORNER_NEIGHBORS[corner_select]
        n1_key, n2_key = f"neighbor_{n1_dir}", f"neighbor_{n2_dir}"

        # Validate neighbors are provided
        n1_image = kwargs.get(n1_key)
        n2_image = kwargs.get(n2_key)
        if n1_image is None or n2_image is None:
            logger.warning(f"[CornerEdges] Missing neighbors for {corner_select} corner: "
                           f"need {n1_dir} and {n2_dir}")
            empty_latent = {"samples": center_latent, "noise_mask": torch.zeros(1, 1, 1, 1)}
            return (empty_latent, torch.zeros(1, 1), *[torch.zeros(1, 1)] * 6,
                    torch.zeros(1, 1, 1, 3), torch.zeros(1, 1, 1, 3),
                    empty_latent, torch.zeros(1, 1, 1, 3), torch.zeros(1, 1), torch.zeros(1, 1))

        n1_latent = neighbor_latents.get(n1_dir)
        n2_latent = neighbor_latents.get(n2_dir)
        if n1_latent is None:
            n1_latent = vae.encode(n1_image)
        if n2_latent is None:
            n2_latent = vae.encode(n2_image)

        # 1. Determine tile priorities (needed before composite)
        tile_priorities = [None, None, None]  # [center, n1, n2]
        if priorities is not None:
            tile_priorities[0] = match_terrain_priority(center_image, priorities)
            tile_priorities[1] = match_terrain_priority(n1_image, priorities)
            tile_priorities[2] = match_terrain_priority(n2_image, priorities)
            logger.info(f"[CornerEdges] Priorities: center={tile_priorities[0]}, "
                         f"{n1_dir}={tile_priorities[1]}, {n2_dir}={tile_priorities[2]}")

        # 2. Composite corner latents in priority order.
        # Lowest priority pasted first, highest priority pasted last —
        # higher-priority tile overwrites at overlapping hex edges.
        # This eliminates the need for a separate edge buffer.
        composited_4d, _ = _normalize_latent(center_latent)
        _, _, H_lat, W_lat = composited_4d.shape
        hex_radius_lat = min(W_lat, H_lat) // 2

        composited = composite_corner_latents(
            center_latent, n1_latent, n2_latent, corner_select, hex_radius_lat,
            tile_priorities=tile_priorities,
        )

        # 3. Generate corner mask at latent resolution
        feather_lat = max(0, round(feather_radius * max(1, int(border_width * hex_radius_lat))))
        border_mask_lat = create_corner_masks(
            W_lat, H_lat, corner_select, border_width, feather_lat, mask_extent,
            tile_priorities,
        )

        # Remap mask strength
        if mask_strength_min > 0.0 or mask_strength_max < 1.0:
            nonzero = border_mask_lat > 0
            border_mask_lat = torch.where(
                nonzero,
                mask_strength_min + border_mask_lat * (mask_strength_max - mask_strength_min),
                border_mask_lat,
            )

        # 4. Build latent dict
        noise_mask = border_mask_lat.unsqueeze(0)
        latent_dict = {"samples": composited, "noise_mask": noise_mask}

        # 5. Generate mask at image resolution (for debug output)
        H_img, W_img = center_image.shape[1], center_image.shape[2]
        hex_radius_img = min(W_img, H_img) // 2
        feather_img = max(0, round(feather_radius * max(1, int(border_width * hex_radius_img))))
        border_mask_img = create_corner_masks(
            W_img, H_img, corner_select, border_width, feather_img, mask_extent,
            tile_priorities,
        )
        if mask_strength_min > 0.0 or mask_strength_max < 1.0:
            nonzero_img = border_mask_img > 0
            border_mask_img = torch.where(
                nonzero_img,
                mask_strength_min + border_mask_img * (mask_strength_max - mask_strength_min),
                border_mask_img,
            )

        # 6. Preview and overlap images
        # Preview: corner composite showing tile arrangement
        corner_preview = composite_corner_preview(center_image, n1_image, n2_image, corner_select, tile_priorities)

        # Overlap: extends neighbor content into border mask area
        corner_overlap = corner_preview.clone()
        border_any = border_mask_img > 0
        if border_any.dim() == 3:
            border_any = border_any[0]
        if border_any.any():
            offsets = get_corner_offsets(corner_select, hex_radius_img)
            _, n1_off, n2_off = offsets

            ys, xs = torch.meshgrid(
                torch.arange(H_img, dtype=torch.long),
                torch.arange(W_img, dtype=torch.long),
                indexing='ij',
            )
            cx, cy = W_img / 2.0, H_img / 2.0

            d_n1 = (xs.float() - (cx + n1_off[0])) ** 2 + (ys.float() - (cy + n1_off[1])) ** 2
            d_n2 = (xs.float() - (cx + n2_off[0])) ** 2 + (ys.float() - (cy + n2_off[1])) ** 2

            use_n1 = border_any & (d_n1 <= d_n2)
            use_n2 = border_any & (d_n2 < d_n1)

            if use_n1.any():
                ox, oy = n1_off
                src_ys = (ys[use_n1] - oy).clamp(0, H_img - 1)
                src_xs = (xs[use_n1] - ox).clamp(0, W_img - 1)
                corner_overlap[0][use_n1] = n1_image[0, src_ys, src_xs]

            if use_n2.any():
                ox, oy = n2_off
                src_ys = (ys[use_n2] - oy).clamp(0, H_img - 1)
                src_xs = (xs[use_n2] - ox).clamp(0, W_img - 1)
                corner_overlap[0][use_n2] = n2_image[0, src_ys, src_xs]

        # 7. Assemble outputs (same count as CentralTile for compatibility)
        outputs = [
            latent_dict,            # LATENT
            border_mask_img,        # MASK (debug)
            *[torch.zeros(H_img, W_img)] * 6,  # 6 directional masks (unused)
            corner_preview if enable_preview else torch.zeros(1, 1, 1, 3, dtype=torch.float32),  # composited_preview
            torch.zeros(1, H_img, W_img, 3),       # color_mask (unused)
            {"samples": center_latent, "noise_mask": noise_mask},  # paintbrush latent
            torch.zeros(1, H_img, W_img, 3),       # paintbrush preview (unused)
            torch.zeros(1, H_lat, W_lat),           # waste_mask_lat (unused)
            torch.zeros(1, H_img, W_img),           # waste_mask_img (unused)
            corner_overlap,                         # overlap_image
        ]

        logger.info(f"[CornerEdges] Total: {time.time()-t_start:.3f}s, "
                    f"corner={corner_select}, extent={mask_extent}")
        return tuple(outputs)
