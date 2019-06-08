"""
Tools to manipulate arrays or tensors.

"""
import torch as t
import numpy as np 

def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()

def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        data= t.from_numpy(data)
    if isinstance(data, t.Tensor):
        data = data.detach()
    
    if cuda:
        data = data.cuda()

    return data

def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.item()