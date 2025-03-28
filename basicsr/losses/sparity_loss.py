
import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F


_reduction_modes = ['none', 'mean', 'sum']

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x


def sparity_loss(pred, target):
    pred_spmask = gumbel_softmax(pred,1,1)
    target_spmask = gumbel_softmax(target,1,1)
    return F.mse_loss(pred_spmask, target_spmask, reduction='none') 