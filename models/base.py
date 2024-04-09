# -*- coding: utf-8 -*-
# ---------------------

from abc import ABCMeta
from abc import abstractmethod
from typing import Union

import torch
from path import Path
from torch import nn


class BaseModel(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        # type: () -> None
        super().__init__()


    def kaiming_init(self, activation):
        # type: (str) -> ()
        """
        Apply "Kaiming-Normal" initialization to all Conv2D(s) of the model.
        :param activation: activation function after conv;
            ->> values in {'relu', 'leaky_relu'}
        """
        assert activation in ['ReLU', 'LeakyReLU', 'leaky_relu'], \
            '`activation` must be \'ReLU\' or \'LeakyReLU\''

        if activation == 'LeakyReLU':
            activation = 'leaky_relu'
        activation = activation.lower()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out',
                    nonlinearity=activation
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    @abstractmethod
    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.

        :param x: input tensor
        """
        ...


    @property
    def n_param(self):
        # type: () -> int
        """
        :return: number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    @property
    def device(self):
        # type: () -> str
        """
        Check the device on which the model is currently located.

        :return: string that represents the device on which the model
            is currently located
            ->> e.g.: 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...
        """
        return str(next(self.parameters()).device)


    @property
    def is_cuda(self):
        # type: () -> bool
        """
        Check if the model is on a CUDA device.

        :return: `True` if the model is on CUDA; `False` otherwise
        """
        return 'cuda' in self.device


    def save_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        Save model weights to the specified path.

        :param path: path of the weights file to be saved.
        """
        torch.save(self.state_dict(), path)


    def load_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        Load model weights from the specified path.

        :param path: path of the weights file to be loaded.
        """
        self.load_state_dict(torch.load(path))


    def requires_grad(self, flag):
        # type: (bool) -> None
        """
        Set the `requires_grad` attribute of all model parameters to `flag`.

        :param flag: True if the model requires gradient, False otherwise.
        """
        for p in self.parameters():
            p.requires_grad = flag
