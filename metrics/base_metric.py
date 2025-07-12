import torch

from abc import abstractmethod


class BaseMetric():
    def __init__(self, name):
        self._num_corrects = 0
        self._num_samples = 0
        self.name = name
        super().__init__()

    def reset(self):
        self._num_corrects = 0
        self._num_samples = 0

    def update(self, model_output, y_true):
        y_pred = torch.argmax(model_output, dim=1)
        # print(f'len y_pred = {len(y_pred)}')
        # print(f'y_pred = {y_pred}')
        # print(f'y_pred shape = {y_pred.shape}')
        # print(f'y_true shape = {y_true.shape}')
        self._num_corrects += torch.sum(y_pred == y_true).item()
        self._num_samples += y_true.shape[0]

    @abstractmethod
    def compute_metric(self):
        pass
