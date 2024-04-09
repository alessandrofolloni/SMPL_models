# -*- coding: utf-8 -*-
# ---------------------

import random
import time
from typing import Tuple

import imgaug
import numpy as np
import torch
from path import Path
from torch.utils.data import Dataset

import mmu
from conf import Conf
from pre_processing import PreProcessor


class DummyDS(Dataset):
    """
    Dataset composed of pairs (x, y) in which:
    * x: RGB image of size 128x128 representing a light blue circle
        (radius = 16 px) on a dark background (random circle position,
        randomly colored dark background)
    * y: copy of x, with the light blue circle surrounded with a red
        line (4 px internal stroke)
    """


    def __init__(self, cnf, mode):
        # type: (Conf, str) -> None
        """
        :param cnf: configuration object
        :param mode: dataset working mode;
            >> values in {'train', 'test'}
        """
        self.cnf = cnf
        self.mode = mode

        ds_dir = self.cnf.ds_root / mode
        self.paths = []
        for x_path in ds_dir.files('*_x.png'):
            y_path = Path(x_path.replace('_x.png', '_y.png'))
            self.paths.append((x_path, y_path))

        self.pre_proc = PreProcessor(unsqueeze=False, device='cpu')


    def __len__(self):
        # type: () -> int
        return len(self.paths)


    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        x_path, y_path = self.paths[i]

        # read input and target
        x_img = mmu.imread(x_path)
        y_img = mmu.imread(y_path)

        # apply pre processing to input and target
        x = self.pre_proc.apply(x_img)
        y = self.pre_proc.apply(y_img)

        return x, y


    @staticmethod
    def wif(worker_id):
        # type: (int) -> None
        """
        Worker initialization function: set random seeds
        :param worker_id: worker int ID
        """
        seed = (int(round(time.time() * 1000)) + worker_id) % (2 ** 32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        imgaug.seed(seed)


    @staticmethod
    def wif_test(worker_id):
        # type: (int) -> None
        """
        Worker initialization function: set random seeds
        :param worker_id: worker int ID
        """
        seed = worker_id + 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        imgaug.seed(seed)


def main():
    ds = DummyDS(cnf=Conf(exp_name='default'), mode='test')

    for i in range(len(ds)):
        x, y = ds[i]
        print(f'Example #{i}: x.shape={x.shape}, y.shape={y.shape}')


if __name__ == '__main__':
    main()
