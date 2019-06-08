import torch

def accuracy(y_pred, gt, topk=(1,)):
    """ Measure accuracy for classification.
    @Args:
        y_pred: Tensor, predictions of the model
        gt: ground truth labels
        topk: tuple, specific top-k evaluation
    Return:
        tuple (tok1, (top-k)) accuracies
    """
    maxk = max(topk)
    batch_size = gt.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(gt.view(1, -1).expand_as(pred))

    res=[]
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def oracle_accuracy(pred_list, target):
    '''
    evaluate oracle error rate
    Args:
        pred_list: selected topk minimal loss indices
        target: true labels
    Return:
        oracle error rate
    '''

    pred_list = [pred.max(1)[1] for pred in pred_list]
    comp_list = [pred.eq(target).float().unsqueeze(1) for pred in pred_list]
    # err_list = [100.0 * (1. - torch.mean(comp)) for comp in comp_list]
    tsum = sum(comp_list)
    tsum = tsum.ge(1).float()

    oracle_acc = 100.0 * torch.mean(tsum)
    return oracle_acc