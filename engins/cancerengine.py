import numpy as np
import torch

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping

from baseengine import BaseEngine

class CancerEngine(BaseEngine):

    def __init__(self, func, model, crit, optimizer, config):
        super(CancerEngine, self).__init__(func, model, crit, optimizer, config)
        pass

    @staticmethod
    def train(self, batch):
        # (batch) = [x, y]
        # |x| = (batch_size, #features)
        # |y| = (batch_size, )
        device = self.config.device

        x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to(device)

        self.model.train()

        y_hat, mask = self.model(x)
        loss = self.crit(y_hat, y)

        loss.backward()
        self.optimizer.step()

        return {
            'loss' : float(loss),
            'mask' : mask.to('cpu').numpy()
        }

    @staticmethod
    def valid(self, batch):
        # (batch) = [x, y]
        # |x| = (batch_size, #features)
        # |y| = (batch_size, )
        device = self.config.device

        x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to(device)

        self.model.eval()

        with torch.no_grad():
            y_hat, mask = self.model(x)
            loss = self.crit(y_hat, y)

        return {
            'loss' : float(loss),
            'mask' : mask.to('cpu').numpy()
        }


