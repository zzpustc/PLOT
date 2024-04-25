import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
import random

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def metric_loss(feas, y):
    """
    calcaute feature metric loss for representations
    """
    y_np = y.detach().cpu().numpy()
    cls_num = np.unique(y_np)
    query_list = []
    support_list = []
    for c in cls_num:
        idx = np.where(y_np == c)[0]
        sel_idx = random.choice(idx)
        rest_idx = [id for id in idx if id != sel_idx]
        query_list.append(feas[sel_idx])
        support_list.append(feas[rest_idx].mean(0))

    labels = torch.arange(len(cls_num)).long().cuda()
    query_list, support_list = torch.stack(query_list), torch.stack(support_list)
    logits = euclidean_metric(query_list, support_list)
    loss = F.cross_entropy(logits, labels)
    return loss


class FocalLoss(nn.Module):
    """
    Implementation of focal loss
    """
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target!=0).type(torch.LongTensor).cuda()
            at = self.alpha.gather(0,select.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)** self.gamma * logpt
        
        return loss


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)