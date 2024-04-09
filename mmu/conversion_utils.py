import os

import matplotlib


# Set matplotlib backend. Use 'Agg' (a non-interactive backend) if no
# display is available (e.g. when running on a remote server); otherwise
# use 'TkAgg', which is the default interactive backend
__DISPLAY = os.environ.get('DISPLAY', None)
if __DISPLAY is None or __DISPLAY == '':
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

from matplotlib import figure
import numpy as np
import torch


def torch2np(x, squeeze=True, is_img=False):
    # type: (torch.Tensor, bool, bool) -> np.ndarray
    """
    Convert a torch tensor to a numpy array.

    :param x: torch tensor to be converted to numpy array
    :param squeeze: whether to remove single-dimensions from the numpy array
    :param is_img: whether the tensor represents an image
        with shape=([B],C,H,W) and values in [0, 1]
    :return: numpy array version of the input torch tensor; if `is_img` is
        `True` then the input image tensor is converted to a numpy image
        np.uint8 array with shape ([B],H,W,C) and values in [0, 255]
    """
    x = x.detach().cpu().numpy()

    if is_img:
        if x.ndim == 4:
            x = x.transpose(0, 2, 3, 1)
        elif x.ndim == 3:
            x = x.transpose(1, 2, 0)
        else:
            raise ValueError(f'[!] invalid image tensor shape: {x.shape}')
        x = (x * 255).astype(np.uint8)

    if squeeze:
        x = x.squeeze()

    return x


def np2torch(x, device, unsqueeze=True):
    # type: (np.ndarray, str, bool) -> torch.Tensor
    """
    Convert a numpy array to a torch tensor.

    :param x: numpy array to be converted to torch tensor
    :param device: device to send the tensor to
    :param unsqueeze: whether to add a single dimension to the tensor
    :return: torch tensor version of the input numpy array
        ->> NOTE: it returns a `float` tensor
    """
    x = torch.from_numpy(x).float().to(device)
    if unsqueeze:
        x = x.unsqueeze(0)
    return x


def pyplot_to_numpy(plt_fig):
    # type: (figure.Figure) -> np.ndarray
    """
    Convert a PyPlot figure into a NumPy array.

    :param plt_fig: figure you want to convert
    :return: converted NumPy array
        ->> shape: (H,W,C); values in [0, 255]
    """
    plt_fig.canvas.draw()
    x = np.fromstring(plt_fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    x = x.reshape(plt_fig.canvas.get_width_height()[::-1] + (3,))
    return x


def pyplot_to_tensor(plt_figure):
    # type: (figure.Figure) -> torch.Tensor
    """
    Convert a PyPlot figure into a torch tensor.

    :param plt_figure: figure you want to convert
    :return: converted torch tensor
        ->> shape: (C,H,W); values in [0, 1]
    """
    np_img = pyplot_to_numpy(plt_figure)
    return np2torch(np_img, device='cuda', unsqueeze=False)
