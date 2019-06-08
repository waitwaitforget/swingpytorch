from __future__ import absolute_import

from independent import IndependentEnsemble

import torchvision.models as models

def test():
    nets = [models.resnet18() for _ in range(3)]
    ide = IndependentEnsemble(nets)
    print(ide.size)

    import torch
    x=torch.rand(10,3,224,224)
    y = torch.LongTensor(10).random_(1000)
    p = ide(x)

    loss = ide.loss_function(p, y)
    print(loss)

if __name__=='__main__':
    test()