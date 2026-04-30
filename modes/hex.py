"""
Hexagonal tiling implementation

Some of this code is taken from excelent guide https://www.redblobgames.com/grids/hexagons/
"""

import math
import functools

from .utils import rotation_matrix
import numpy as np


def cube_to_axial(cube_coords: tuple[int, int, int]) -> tuple[int, int]:
    """
    Convert cube coordinates to axial coordinates

    :param cube_coords: Cube coordinates
    :return: Axial coordinates
    """

    return (cube_coords[0], cube_coords[1])


def axial_to_cube(axial_coords: tuple[int, int]) -> tuple[int, int, int]:
    """
    Convert axial coordinates to cube coordinates

    :param axial_coords: Axial coordinates
    :return: Cube coordinates
    """

    q = axial_coords[0]
    r = axial_coords[1]
    s = -q - r

    return (q, r, s)


def axial_round(frac_coords: tuple[float, float]) -> tuple[int, int]:
    """
    Round fractional axial coordinates to nearest axial coordinate

    :param frac_coords: Fractional axial coordinates
    :return: Axial coordinates
    """

    return cube_to_axial(cube_round(axial_to_cube(frac_coords)))


def cube_round(frac_coords: tuple[float, float, float]) -> tuple[int, int, int]:
    """
    Round fractional cube coordinates to nearest cube coordinate

    :param frac_coords: Fractional cube coordinates
    :return: Cube coordinates
    """

    q = math.floor(frac_coords[0] + 0.5)
    r = math.floor(frac_coords[1] + 0.5)
    s = math.floor(frac_coords[2] + 0.5)

    q_diff = abs(q - frac_coords[0])
    r_diff = abs(r - frac_coords[1])
    s_diff = abs(s - frac_coords[2])

    if q_diff > r_diff and q_diff > s_diff:
        q = -r - s
    elif r_diff > s_diff:
        r = -q - s
    else:
        s = -q - r

    return (q, r, s)


@functools.cache
def get_matrix(rotation: float) -> np.ndarray:
    """
    Get rotation matrix

    :param rotation: Rotation angle in degrees
    :return: Rotation matrix
    """

    return np.matmul(
        rotation_matrix(rotation),
        # Hexagon basis vectors
        np.array([[math.sqrt(3), math.sqrt(3) / 2], [0, 3 / 2]]),
    )


@functools.cache
def get_inverse_matrix(rotation: float) -> np.ndarray:
    """
    Get inverse rotation matrix

    :param rotation: Rotation angle in degrees
    :return: Inverse rotation matrix
    """

    return np.linalg.inv(get_matrix(rotation))


def hex_to_pixel(
    hex_coords: tuple[int, int], size: int, rotation: float
) -> tuple[int, int]:
    """
    Convert hexagonal coordinates to pixel coordinates

    :param hex_coords: Hexagonal coordinates
    :param size: Size of hexagon
    :param rotation: Rotation angle in degrees
    :return: Pixel coordinates
    """

    (x, y) = (
        size
        * np.matmul(
            get_matrix(rotation),
            np.array([[hex_coords[0]], [hex_coords[1]]]),
        ).flatten()
    )

    return (int(math.floor(x + 0.5)), int(math.floor(y + 0.5)))


def pixel_to_hex(
    pixel_coords: tuple[int, int], size: int, rotation: float
) -> tuple[float, float]:
    """
    Convert pixel coordinates to fractional hexagonal coordinates

    :param pixel_coords: Pixel coordinates
    :param size: Size of hexagon
    :param rotation: Rotation angle in degrees
    :return: Fractional hexagonal coordinates
    """

    (q, r) = (
        np.matmul(
            get_inverse_matrix(rotation),
            np.array([[pixel_coords[0]], [pixel_coords[1]]]),
        ).flatten()
        / size
    )

    return (q, r)


@functools.cache
def hex_tiling(
    x: int,
    y: int,
    padded_size: tuple[int, int],
    hex_size: float,
    rotation: float,
) -> tuple[int, int]:
    """Hexagonal tiling with pre-computed hex radius.

    :param x: X coordinate in padded space
    :param y: Y coordinate in padded space
    :param padded_size: (width, height) of padded tensor
    :param hex_size: Hex radius (pre-computed at appropriate resolution)
    :param rotation: Rotation angle in degrees
    :return: (new_x, new_y) source coordinates in padded space
    """
    q, r = pixel_to_hex(
        (x - padded_size[0] // 2, y - padded_size[1] // 2),
        hex_size,
        rotation,
    )
    rounded = axial_round((q, r))
    q -= rounded[0]
    r -= rounded[1]
    new_x, new_y = hex_to_pixel((q, r), hex_size, rotation)
    new_x = (new_x + padded_size[0] // 2) % padded_size[0]
    new_y = (new_y + padded_size[1] // 2) % padded_size[1]

    return (new_x, new_y)


def hex_tiling_at(x, y, padded_size, resolved):
    """Hexagonal tiling with automatic resolution scaling from ResolvedSettings.

    Scales hex size to the tensor resolution of padded_size,
    then delegates to hex_tiling().
    """
    size = resolved.hex_size_at(padded_size[0], padded_size[1])
    return hex_tiling(x, y, padded_size, size, resolved.rotation)
