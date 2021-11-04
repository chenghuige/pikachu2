import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def onehot_embedding(labels,num_classes):
    N = labels.size(0)
    D = num_classes
    y = torch.zeros(N,D)
    y[torch.arange(0,N).long(),labels] = 1
    return y

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def focal_loss2d(input, target, start_cls_index=1,size_average=True):
    n, c, h, w = input.size()
    p = F.softmax(input)
    p = p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    p = p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= start_cls_index] #exclude background example
    p = p.view(-1, c)

    mask = target >=start_cls_index #exclude background example
    target = target[mask]

    t = onehot_embedding(target.data.cpu(),c)
    t = Variable(t).cuda()

    alpha = 0.25
    gamma = 2
    w = alpha* t + (1-alpha)*(1-t)
    w = w * (1-p).pow(gamma)

    loss = F.binary_cross_entropy(p,t,w,size_average=False)

    if size_average:
       loss /= mask.data.sum()
    return loss

def bin_clsloss(input, target, size_average=True):
    n, c = input.size()
    p = input
    target_emdding=torch.zeros((n,c))
    for i in range(n):
        nclasses = set(target.data.cpu().numpy()[i].flat)
        for nclass in nclasses:
            target_emdding[i][nclass]=1.0

    mask = target >= 0

    t = target_emdding[:,1:] #exclude background
    t = Variable(t).cuda()

    p = p[:,1:] #exclude background
    p = F.sigmoid(p) #binaray cls

    loss = F.binary_cross_entropy(p,t,size_average=size_average)

    return loss