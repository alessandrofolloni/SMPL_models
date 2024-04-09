# -*- coding: utf-8 -*-
# ---------------------

import os


# add the project root to the PYTHONPATH to allow absolute imports
# ->> NOTE: PyCharm does this automatically, but it is not the case if
#     you run the code from the terminal.
PYTHONPATH = '..:.'
if os.environ.get('PYTHONPATH', default=None) is None:
    os.environ['PYTHONPATH'] = PYTHONPATH
else:
    os.environ['PYTHONPATH'] += (':' + PYTHONPATH)

import yaml
import socket
import random
import torch
import numpy as np
from path import Path
from typing import Optional
import termcolor
from typing import Dict
from typing import Any


def set_seed(seed=None):
    # type: (Optional[int]) -> int
    """
    Set the random seed using the required value (`seed`)
    or a random value if `seed` is `None`.
    :return: the newly set seed
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


class Conf(object):
    HOSTNAME = socket.gethostname()


    @property
    def dict_view(self):
        # type: () -> Dict[str, Any]
        """
        :return: dictionary version of the configuration file
        """
        x = self.__dict__
        y = {}
        for key in x:
            if key not in self.keys_to_hide:
                y[key] = x[key]
        return y


    def __init__(self, cnf_path=None, seed=None, exp_name=None, log=True):
        # type: (str, int, str, bool) -> None
        """
        :param cnf_path: optional path of the configuration file
        :param seed: desired seed for the RNG;
            >> if `None`, it will be chosen randomly
        :param exp_name: name of the experiment
        :param log: `True` if you want to log each step; `False` otherwise
        """
        self.exp_name = exp_name
        self.log_each_step = log

        # print project name and host name
        project_name = Path(__file__).parent.parent.basename()
        m_str = f'┃ {project_name}@{Conf.HOSTNAME} ┃'
        u_str = '┏' + '━' * (len(m_str) - 2) + '┓'
        b_str = '┗' + '━' * (len(m_str) - 2) + '┛'
        print(u_str + '\n' + m_str + '\n' + b_str)

        # define output paths
        self.project_root = Path(__file__).parent.parent
        self.project_log_path = self.project_root.abspath() / 'log'
        self.exp_log_path = self.project_log_path / exp_name

        # set random seed
        self.seed = set_seed(seed)  # type: int

        self.keys_to_hide = list(self.__dict__.keys()) + ['keys_to_hide']

        # if the configuration file is not specified
        # try to load a configuration file based on the experiment name
        tmp = Path(__file__).parent / (self.exp_name + '.yaml')
        if cnf_path is None and tmp.exists():
            cnf_path = tmp

        # read the YAML configuration file
        if cnf_path is None:
            y = {}
        else:
            conf_file = open(cnf_path, 'r')
            y = yaml.load(conf_file, Loader=yaml.Loader)

        # read configuration parameters from YAML file
        # or set their default value
        self.lr = y.get('LR', 0.0001)  # type: float
        self.epochs = y.get('EPOCHS', 32)  # type: int
        self.n_workers = y.get('N_WORKERS', 4)  # type: int
        self.batch_size = y.get('BATCH_SIZE', 8)  # type: int
        self.max_patience = y.get('MAX_PATIENCE', 8)  # type: int

        default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = y.get('DEVICE', default_device)  # type: str

        self.ds_root = y.get('DS_ROOT', None)  # type: str

        assert self.ds_root, \
            f'you must specify the `DS_ROOT` parameter ' \
            f'in the configuration file'

        self.ds_root = self.ds_root.replace(
            '$PROJECT_DIR', self.project_root
        )
        self.ds_root = Path(self.ds_root)
        assert self.ds_root.exists(), \
            f'directory `DS_ROOT={self.ds_root}` does not exist'


    def write_to_file(self, out_file_path):
        # type: (str) -> None
        """
        Writes configuration parameters to `out_file_path`
        :param out_file_path: path of the output file
        """
        import re

        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        text = ansi_escape.sub('', str(self))
        with open(out_file_path, 'w') as out_file:
            print(text, file=out_file)


    def __str__(self):
        # type: () -> str
        """
        :return: string representation of the configuration object;
            ->> NOTE: this is a color-coded string
        """
        out_str = ''
        for key in self.__dict__:
            if key in self.keys_to_hide:
                continue
            value = self.__dict__[key]
            if type(value) is Path or type(value) is str:
                value = termcolor.colored(value, 'yellow')
            else:
                value = termcolor.colored(f'{value}', 'magenta')
            out_str += termcolor.colored(f'{key.upper()}', 'blue')
            out_str += termcolor.colored(': ', 'red')
            out_str += value
            out_str += '\n'
        return out_str[:-1]


    def no_color_str(self):
        # type: () -> str
        """
        :return: string representation of the configuration object;
            ->> NOTE: this is NOT a color-coded string
        """
        out_str = ''
        for key in self.__dict__:
            value = self.__dict__[key]
            out_str += f'{key.upper()}: {value}\n'
        return out_str[:-1]


def show_default_params():
    """
    Print default configuration parameters
    """
    cnf = Conf(exp_name='default')
    print(f'\nDefault configuration parameters: \n{cnf}')


if __name__ == '__main__':
    show_default_params()
