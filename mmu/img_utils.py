from typing import Optional
from typing import Tuple

import cv2
import numpy as np
from matplotlib import cm
from torch import Tensor
from torchvision.transforms import ToTensor


def imread(img_path):
    # type: (str) -> Optional[np.ndarray]
    """
    Reads an image ('RGB' order) from the specified file and returns it.
    If the image cannot be read (because of missing file, improper
    permissions, unsupported or invalid format), the function returns None

    :param img_path: path of the image to read
    :return: image as numpy array with shape (H,W,C) and values in [0,255]
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite(out_path, img):
    # type: (str, np.ndarray) -> None
    """
    Saves image `img` to the specified file. The image format is chosen
    based on the filename extension.
    >> NOTE: In general, only np.uint8 single-channel or 3-channel
       (with 'RGB' order) images can be saved using this function.

    :param out_path: path of the output file
    :param img: image you want to save
    :return: `True` if the image has been saved correctly, `False` otherwise
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(out_path, img)


def imshow(img, waitkey=None, header='imshow'):
    # type: (np.ndarray, Optional[int], str) -> int
    """
    Displays the input image `img` ('RGB' order) in a window with title
    `header` and waits for a user key event.

    :param img: image ('RGB' order) you want to show;
        >> shape: (H,W) or (H,W,3) and values in [0, 255]
    :param waitkey: milliseconds you want to wait;
        >> if `None` it waits indefinitely
    :param header: title of the window that displays the image
    :return: int value of the pressed key
    """

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(header, img_bgr)

    key = -1
    if waitkey is None:
        while cv2.getWindowProperty(header, 0) >= 0 and key == -1:
            key = cv2.waitKey(30)
    elif waitkey > 0:
        key = cv2.waitKey(waitkey)

    return key


def apply_colormap_to_tensor(x, cmap='jet', value_range=(None, None)):
    # type: (Tensor, str, Optional[Tuple[float, float]]) -> Tensor
    """
    Apply a color map to an input tensor representing a 1-channel image.

    :param x: Tensor with shape (1, H, W)
    :param cmap: name of the matplotlib color map you want to apply;
        ->> see https://matplotlib.org/stable/tutorials/colors/colormaps.html
            for a list of available color maps
    :param value_range: (min, max) value range for tensor `x`
    :return: Tensor with shape (3, H, W)
    """
    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(vmin=value_range[0], vmax=value_range[1])
    x = x.detach().cpu().numpy()
    x = x.squeeze()
    x = cmap.to_rgba(x)[:, :, :-1]
    return ToTensor()(x)
