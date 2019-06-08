import torch as t
import shutil

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Save checkpoints for current state.
    Args:
        state: dict, state dict to store to disk
        is_best: boolean, is the best evaluation model
        filename: str
    
    """
    t.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')