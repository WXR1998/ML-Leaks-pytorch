from typing import Optional
import torch
from torch.utils.data import DataLoader
import os
import logging
import tqdm
import numpy as np
from torch.nn.functional import one_hot

import nn
import util
import configs
import dataset


def categorical_crossentropy(pred, label):
    label = one_hot(label, 10)
    return torch.sum(-label * torch.log(pred), dim=1)

class Model:
    def __init__(self,
                 epochs: int=100,
                 lr: float=0.001,
                 dataset_type: util.Dataset=util.Dataset.MNIST,
                 device: torch.device=torch.device('cpu')):

        self._model: Optional[torch.nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._model_type = None

        self._epochs = epochs
        self._lr = lr
        self._dataset_type = dataset_type
        self._device = device

        self._loss = categorical_crossentropy

    def dump(self, name: str):
        state = self._model.state_dict()
        os.makedirs(configs.ckpt_path, exist_ok=True)
        torch.save(state, os.path.join(configs.ckpt_path, f'{self._model_type.name}_{name}.pt'))

    def load(self, name: str):
        os.makedirs(configs.ckpt_path, exist_ok=True)
        ckpt_path = os.path.join(configs.ckpt_path, f'{self._model_type.name}_{name}.pt')

        if not os.path.exists(ckpt_path):
            logging.error(f'Ckpt file "{ckpt_path}" does not exist.')
            exit(0)

        ckpt = torch.load(ckpt_path)
        self._model.load_state_dict(ckpt)

    def train(self, train_loader: DataLoader):
        logging.info('Start training...')

        for epoch in range(self._epochs):
            loss_sum = 0
            acc_sum = 0
            for batch_x, batch_y in tqdm.tqdm(train_loader):
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                self._optimizer.zero_grad()
                pred = self._model(batch_x)
                loss = torch.mean(self._loss(pred, batch_y))
                loss.backward()
                self._optimizer.step()

                pred_v = np.argmax(pred.detach().cpu().numpy(), axis=-1)
                batch_y_v = batch_y.detach().cpu().numpy()
                loss_sum += loss.cpu().item() * batch_x.shape[0]
                acc_sum += sum(pred_v == batch_y_v)

            sample_count = len(train_loader.dataset)
            loss_avg = loss_sum / sample_count
            acc_avg = acc_sum / sample_count
            logging.info(f'Epoch {epoch + 1}/{self._epochs}:\tloss = {loss_avg:.3f}\tacc = {acc_avg:.3f}')

            # if acc_avg >= 0.999:
            #     logging.info('Acc threshold arrived, early stop.')
            #     break

        logging.info('Finish training.')

    def inference(self, test_loader: dataset.DataLoader):
        """
        :param test_loader:
        :return: The predicted posteriors
        """
        logging.info('Start inference...')
        preds = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self._device)
                pred = self._model(batch_x)
                preds.append(pred.detach().cpu().numpy())

        logging.info('Finish inference.')

        preds = np.concatenate(preds, axis=0)
        return preds

class ShadowModel(Model):
    def __init__(self, **kwargs):
        super(ShadowModel, self).__init__(**kwargs)
        self._model_type = util.ModelType.SHADOW

        self._model = nn.Conv(self._dataset_type).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=1e-5)

class TargetModel(Model):
    def __init__(self, **kwargs):
        super(TargetModel, self).__init__(**kwargs)
        self._model_type = util.ModelType.TARGET

        self._model = nn.Conv(self._dataset_type).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=1e-5)

class AttackModel(Model):
    def __init__(self, **kwargs):
        super(AttackModel, self).__init__(**kwargs)
        self._model_type = util.ModelType.ATTACK

        self._model = nn.Attack().to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        self._loss = torch.nn.CrossEntropyLoss()

    def predict(self, test_loader: dataset.DataLoader):
        p = self.inference(test_loader)
        return np.argmax(p, axis=-1)
