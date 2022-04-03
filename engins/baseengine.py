import numpy as np
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping

class BaseEngine(Engine):

    def __init__(self, func, model, crit, optimizer, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func)

        self.best_loss = np.inf
        self.best_model = None

    @staticmethod
    def train(self, batch):
        pass

    @staticmethod
    def valid(self, batch):
        pass

    @staticmethod
    def visual_result_attach(train_engine, valid_engine, train_metrics, valid_metrics, config):
        def attach_running_average(engine, metric):
            RunningAverage(output_transform=lambda x:x[metric]).attach(engine, metric)

        for metric in train_metrics:
            attach_running_average(train_engine, metric)

        for metric in valid_metrics:
            attach_running_average(valid_engine, metric)

        def attach_pbar(engine, metrics):
            pbar = ProgressBar
            pbar.attach(engine, metrics)

        for engine, metrics in zip([train_engine, valid_engine], [train_metrics, valid_metrics]):
            attach_pbar(engine, metrics)


