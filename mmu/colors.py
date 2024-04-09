from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from PIL import ImageColor


Number = Union[int, float]
Color3 = Union[List[Number], Tuple[Number, Number, Number], np.ndarray]

# --- flat colors ---
FLAT_RED = (231, 76, 60)
FLAT_ORANGE = (230, 126, 34)
FLAT_YELLOW = (241, 196, 15)
FLAT_GREEN = (46, 204, 113)
FLAT_BLUE = (52, 152, 219)
FLAT_PURPLE = (155, 89, 182)
FLAT_BLACK = (44, 62, 80)
FLAT_GREY = (149, 165, 166)

# --- GoatAI base colors ---
GOAT_RED = (251, 49, 103)
GOAT_BLUE = (0, 32, 62)
GOAT_WHITE = (233, 237, 243)


def to_color(color, colorspace='RGB'):
    # type: (Union[str, Color3], str) -> tuple[int, int, int]
    """
    Returns a color tuple of 3 integers in range [0, 255].

    :param color: it can be an RGB color sequence of an HEX string
    :param colorspace: colorspace of the output tuple; it can be 'RGB' or 'BGR
    :return: tuple of 3 integers in range [0, 255] representing the input color
    """
    # hex color
    if type(color) is str and color.startswith('#'):
        rgb_color = ImageColor.getcolor(color, 'RGB')
        if colorspace == 'BGR':
            rgb_color = rgb_color[::-1]
    else:
        rgb_color = [int(round(c)) for c in color]
    return tuple(rgb_color)


def demo():
    import mmu

    hex_color = '#FB3167'
    rgb_color = to_color(color=hex_color)
    print(f'$> HEX color: {hex_color}')
    print(f'$> RGB color: {rgb_color}')

    img = np.zeros((256, 256, 3)).astype(np.uint8)
    for channel in range(3):
        img[..., channel] += rgb_color[channel]
    mmu.imshow(img, header=f'RGB={rgb_color}, HEX={hex_color}')


if __name__ == '__main__':
    demo()
