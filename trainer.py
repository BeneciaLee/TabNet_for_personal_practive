import torch
import torch.nn
import numpy as np

from engins.cancerengine import CancerEngine

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping

from copy import deepcopy

class Trainer():

    def __init__(self, config):
        self.config = config

    def train(self, model, crit, optimizer, train_loader, valid_loader = None):
        if train_loader is None:
            print("Train_loader is not configured.")
            raise NotImplementedError

        train_metrics = ['loss']
        valid_metrics = ['loss']

        train_engine = CancerEngine(CancerEngine.train, model, crit, optimizer, self.config)

        if valid_loader is None:
            pass

        valid_engine = CancerEngine(CancerEngine.valid, model, crit, optimizer, self.config)

        CancerEngine.visual_result_attach(train_engine, valid_engine, train_metrics, valid_metrics, self.config)

        @train_engine.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            print(f"Training Results - Epoch[{trainer.state.epoch}/{trainer.state.max_epochs}] Loss : {trainer.state.metrics['loss']}")

        def run_valid(engine, valid_engine, valid_loader):
            valid_engine.run(valid_loader, max_epochs = 1)

        train_engine.add_event_handler(Events.EPOCH_COMPLETED, run_valid, valid_engine, valid_loader)

        @valid_engine.on(Events.EPOCH_COMPLETED)
        def check(engine):
            loss = float(engine.state.metrics['loss'])
            if loss <= engine.best_loss:
                engine.best_loss = loss
                engine.best_model = deepcopy(engine.model.state_dict())

        def score_function(engine):
            val_loss = engine.state.metrics['loss']
            return -val_loss

        handler = EarlyStopping(patience=self.config.patience, score_function=score_function, trainer=train_engine)
        valid_engine.add_event_handler(Events.COMPLETED, handler)

        train_engine.run(train_loader, max_epochs=self.config.max_epochs)


