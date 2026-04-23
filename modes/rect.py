"""
Rectangular (toroidal) tiling implementation

Wraps coordinates modularly: right edge wraps to left, bottom wraps to top.
"""

from . import Settings


def rect_tiling(
    x: int,
    y: int,
    original_size: tuple[int, int],
    padded_size: tuple[int, int],
    _settings: Settings,
) -> tuple[int, int]:
    """
    Rectangular tiling: wraps coordinates modularly around the original area.

    Positions inside original_size are identity. Positions in the padding area
    wrap to the opposite side of the original content.

    :param x: X coordinate in padded space
    :param y: Y coordinate in padded space
    :param original_size: (width, height) of original content
    :param padded_size: (width, height) of padded tensor
    :param _settings: Tiling settings (unused for rectangular)
    :return: (new_x, new_y) source coordinates in padded space
    """

    ow, oh = original_size
    pw, ph = padded_size
    pad_x = (pw - ow) // 2
    pad_y = (ph - oh) // 2

    rel_x = (x - pad_x) % ow
    rel_y = (y - pad_y) % oh

    return (rel_x + pad_x, rel_y + pad_y)
