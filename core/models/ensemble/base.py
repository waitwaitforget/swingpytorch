"""
Basic ensemble network.

"""
import torch.nn as nn

class BasicEnsemble(nn.Module):
    def __init__(self, nets):
        super(BasicEnsemble, self).__init__()

        self.models = nn.ModuleList(nets)
        self.size = len(nets)
        self.criterion = None

    def forward(self, inputs):
        preds = [model(x) for x, model in zip(inputs, self.models)]
        return preds

    def loss_function(self, preds, target):
        raise NotImplementedError("This is a base module, you need to inherit a new module from this class.")

    def cuda(self, devices='cuda'):
        self.models.cuda(devices)
        self.criterion.cuda(devices)