"""
Terrain priority settings for hex inpainting.
Chainable node mapping reference images to priority integers.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class TerrainPriorities:
    """Reference images paired with priority values. Chainable across nodes."""
    entries: list[tuple[torch.Tensor, int]] = field(default_factory=list)


class TerrainPrioritySettings:
    """
    Maps reference terrain images to priority integers.
    Chain multiple nodes for more than 8 terrain types.
    Lower priority index = higher priority (e.g., ShallowWater=1 > Forest=5).
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {},
            "optional": {
                "priorities_in": ("TERRAIN_PRIORITIES", {
                    "tooltip": "Chain input from previous TerrainPrioritySettings node.",
                }),
            },
        }
        for i in range(1, 9):
            inputs["optional"][f"image_{i}"] = ("IMAGE", {
                "tooltip": f"Reference image for terrain type {i}.",
            })
            inputs["required"][f"priority_{i}"] = ("INT", {
                "default": -1,
                "min": -1,
                "max": 100,
                "tooltip": f"Priority for terrain {i}. -1 = unused slot. Lower = higher priority.",
            })
        return inputs

    RETURN_TYPES = ("TERRAIN_PRIORITIES",)
    RETURN_NAMES = ("TERRAIN_PRIORITIES",)
    FUNCTION = "build"
    CATEGORY = "conditioning"

    def build(self, priority_1=-1, priority_2=-1, priority_3=-1, priority_4=-1,
              priority_5=-1, priority_6=-1, priority_7=-1, priority_8=-1, **kwargs):
        # Start from chain input if provided
        entries = []
        priorities_in = kwargs.get("priorities_in")
        if priorities_in is not None:
            entries.extend(priorities_in.entries)

        for i in range(1, 9):
            img = kwargs.get(f"image_{i}")
            pri = locals()[f"priority_{i}"]
            if img is not None and pri >= 0:
                entries.append((img, pri))

        return (TerrainPriorities(entries=entries),)


def match_terrain_priority(
    tile_image: torch.Tensor,
    priorities: TerrainPriorities,
) -> int | None:
    """
    Match a tile image against reference images and return its priority.

    Uses MSE between the tile and each reference image.
    Returns None if no match found (all MSE above threshold).

    :param tile_image: Tile image tensor (1, H, W, 3)
    :param priorities: Terrain priority settings
    :return: Priority int or None
    """
    if not priorities.entries:
        return None

    best_mse = float("inf")
    best_priority = None

    for ref_image, priority in priorities.entries:
        # Handle size mismatch by resizing reference to match tile
        if ref_image.shape != tile_image.shape:
            ref_resized = torch.nn.functional.interpolate(
                ref_image.permute(0, 3, 1, 2),
                size=(tile_image.shape[1], tile_image.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        else:
            ref_resized = ref_image

        mse = ((tile_image.float() - ref_resized.float()) ** 2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best_priority = priority

    # Threshold: if best MSE is too high, treat as unknown
    if best_mse > 0.1:
        return None

    return best_priority
