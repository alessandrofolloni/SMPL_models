from time import time

import numpy as np
import torch
import torchvision as tv
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import Conf
from dataset.dummy_ds import DummyDS
from models import DummyModel
from progress_bar import ProgressBar
from scheduler import LRScheduler


class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> None
        """
        :param cnf: configuration object that contains all the
            hyper-parameters and paths.
        """

        self.cnf = cnf

        # init model
        self.model = DummyModel()
        self.model = self.model.to(cnf.device)

        # init optimizer
        self.optimizer = optim.AdamW(
            params=self.model.parameters(), lr=cnf.lr
        )

        # init train loader
        training_set = DummyDS(cnf, mode='train')
        self.train_loader = DataLoader(
            dataset=training_set, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=True, pin_memory=True,
            worker_init_fn=training_set.wif,
        )

        # init test loader
        test_set = DummyDS(cnf, mode='test')
        self.test_loader = DataLoader(
            dataset=test_set, batch_size=cnf.batch_size,
            num_workers=1, shuffle=False, pin_memory=True,
            worker_init_fn=test_set.wif_test,
        )

        # init learning rate scheduler
        self.scheduler = LRScheduler(
            optimizer=self.optimizer,
            max_lr=cnf.lr, final_lr=(cnf.lr / 100),
            epochs=cnf.epochs, dl_len=len(self.train_loader),
            warmup=int(round(cnf.epochs * 0.03))
        )

        # init logging stuffs
        self.log_path = cnf.exp_log_path
        print(f'tensorboard --logdir={cnf.project_log_path.abspath()}\n')
        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)

        # starting values
        self.epoch = 0
        self.best_test_loss = None
        self.patience = self.cnf.max_patience

        # init progress bar
        self.progress_bar = ProgressBar(
            max_step=self.log_freq, max_epoch=self.cnf.epochs
        )

        # possibly load checkpoint
        self.load_ck()


    def load_ck(self):
        """
        Load training checkpoint from the log directory.
        If no checkpoint is found, nothing happens.
        """
        ck_path = self.log_path / 'training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path)
            print(f'[loading checkpoint \'{ck_path}\']')
            self.epoch = ck['epoch']
            self.progress_bar.current_epoch = self.epoch
            self.model.load_state_dict(ck['model'])
            self.optimizer.load_state_dict(ck['optimizer'])
            self.scheduler.load_state_dict(ck['scheduler'])
            self.scheduler.optimizer = self.optimizer
            self.best_test_loss = ck['best_test_loss']
            self.patience = ck['patience']


    def save_ck(self):
        """
        Save training checkpoint to the log directory.
        """
        ck = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_test_loss': self.best_test_loss,
            'patience': self.patience,
        }
        torch.save(ck, self.log_path / 'training.ck')


    def train(self):
        """
        Train model for one epoch on the training-set.
        """
        start_time = time()
        self.model.train()

        times = []
        train_losses = []
        for step, sample in enumerate(self.train_loader):
            t = time()

            self.optimizer.zero_grad()

            x, y_true = sample
            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)

            y_pred = self.model.forward(x)
            loss = nn.MSELoss()(y_pred, y_true)
            loss.backward()
            train_losses.append(loss.item())

            # update scheduler and optimizer
            self.scheduler.step(step=step, epoch=self.epoch)
            self.optimizer.step(None)

            # print progress bar
            times.append(time() - t)
            c1 = self.cnf.log_each_step
            c2 = (not c1 and self.progress_bar.progress == 1)
            if c1 or c2:
                lr = self.optimizer.param_groups[0]['lr']
                print(f'\r{self.progress_bar} '
                      f'│ Loss: {np.mean(train_losses):.6f} '
                      f'│ LR: {lr:06f}', end='')
            self.progress_bar.inc()

        # log average loss of this epoch
        mean_epoch_loss = np.mean(train_losses)
        self.sw.add_scalar(
            tag='train_loss', scalar_value=mean_epoch_loss,
            global_step=self.epoch
        )

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')


    def test(self):
        """
        Test model on the test-set.
        """
        self.model.eval()

        t = time()
        test_losses = []
        for step, sample in enumerate(self.test_loader):
            x, y_true = sample
            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)
            y_pred = self.model.forward(x)

            loss = nn.MSELoss()(y_pred, y_true)
            test_losses.append(loss.item())

            # draw results for this step in a 3 rows grid:
            # row #1: input (x)
            # row #2: predicted_output (y_pred)
            # row #3: target (y_true)
            grid = torch.cat([x, y_pred, y_true], dim=0)
            grid = tv.utils.make_grid(
                grid, normalize=True, value_range=(0, 1),
                nrow=x.shape[0]
            )
            self.sw.add_image(
                tag=f'results_{step}',
                img_tensor=grid, global_step=self.epoch
            )

        # save best model
        mean_test_loss = np.mean(test_losses)
        first_time = self.best_test_loss is None
        if first_time or (mean_test_loss < self.best_test_loss):
            self.best_test_loss = mean_test_loss
            self.patience = self.cnf.max_patience
            torch.save(self.model.state_dict(), self.log_path / 'best.pth')
        else:
            self.patience = self.patience - 1

        # log test results
        print(f'\t● AVG Loss on TEST-set: {mean_test_loss:.6f}'
              f' │ patience: {self.patience}'
              f' │ T: {time() - t:.2f} s')

        # log test loss / test metric
        self.sw.add_scalar(
            'test_loss', scalar_value=mean_test_loss,
            global_step=self.epoch
        )

        # log patience
        self.sw.add_scalar(
            'patience', scalar_value=self.patience,
            global_step=self.epoch
        )

        if self.patience == 0:
            print('\n--------')
            print(f'[■] Done! -> `patience` reached 0.')
            self.save_ck()
            exit(0)


    def run(self):
        """
        Start model training procedure:
            ->> train ->> test ->> checkpoint ->> repeat
        """
        for _ in range(self.epoch, self.cnf.epochs):
            self.train()

            with torch.no_grad():
                self.test()

            self.epoch += 1
            self.save_ck()
