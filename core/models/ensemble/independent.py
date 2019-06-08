import torch.nn as nn
from base import BasicEnsemble


class IndependentEnsemble(BasicEnsemble):
    def __init__(self, nets):
        super(IndependentEnsemble, self).__init__(nets)

        self.criterion = nn.CrossEntropyLoss()

    def loss_function(self, preds, target):
        return sum([self.criterion(pred, target) for pred in preds])

    