#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ops.py
#        \author   chenghuige  
#          \date   2018-09-30 21:44:39.920227
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable

import gezi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://github.com/rdipietro/pytorch/blob/4c00324affb8c6d53d4362e321ea0e99ede6cfde/torch/nn/utils/rnn.py
def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError('inputs is incompatible with lengths.')
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
            for length in lengths]
    ind = Variable(torch.LongTensor(ind).transpose(0, 1))
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs

#https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)
  
def second_order_fm(x_all):
    summed_emb = torch.sum(x_all, 1)
    summed_emb_square = summed_emb ** 2

    squared_emb = x_all ** 2
    squared_sum_emb = torch.sum(squared_emb, 1) 

    # [None * K]
    y_second_order = 0.5 * (summed_emb_square - squared_sum_emb) 
    return y_second_order   

# https://gist.github.com/bbrighttaer/207dc03b178bbd0fef8d1c0c1390d4be

def segment_sum(data, segment_ids):
    """
    Analogous to tf.segment_sum (https://www.tensorflow.org/api_docs/python/tf/math/segment_sum).

    :param data: A pytorch tensor of the data for segmented summation.
    :param segment_ids: A 1-D tensor containing the indices for the segmentation.
    :return: a tensor of the same type as data containing the results of the segmented summation.
    """
    if not all(segment_ids[i] <= segment_ids[i + 1] for i in range(len(segment_ids) - 1)):
        raise AssertionError("elements of segment_ids must be sorted")

    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError("segment_ids should be the same size as dimension 0 of input.")

    # t_grp = {}
    # idx = 0
    # for i, s_id in enumerate(segment_ids):
    #     s_id = s_id.item()
    #     if s_id in t_grp:
    #         t_grp[s_id] = t_grp[s_id] + data[idx]
    #     else:
    #         t_grp[s_id] = data[idx]
    #     idx = i + 1
    #
    # lst = list(t_grp.values())
    # tensor = torch.stack(lst)

    num_segments = len(torch.unique(segment_ids))
    return unsorted_segment_sum(data, segment_ids, num_segments)

# def unsorted_segment_sum(data, segment_ids, num_segments):
#     """
#     Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

#     :param data: A tensor whose segments are to be summed.
#     :param segment_ids: The segment indices tensor.
#     :param num_segments: The number of segments.
#     :return: A tensor of same data type as the data argument.
#     """
#     assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

#     # segment_ids is a 1-D tensor repeat it to have the same shape as data
#     if len(segment_ids.shape) == 1:
#         s = torch.prod(torch.tensor(data.shape[1:])).long()
#         segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

#     assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

#     shape = [num_segments] + list(data.shape[1:])
#     tensor = torch.zeros(*shape).scatter_add(0, segment_ids, data.float())
#     tensor = tensor.type(data.dtype)
#     return tensor


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    segment_ids = torch.repeat_interleave(segment_ids.unsqueeze(-1), repeats=data.shape[-1], dim=-1)

    shape = [data.shape[0], num_segments] + list(data.shape[2:])
    device_ = gezi.get('device') or device
    tensor = torch.zeros(*shape, device=device_).scatter_add(1, segment_ids, data)
    # tensor = torch.zeros(*shape).cuda().scatter_add(1, segment_ids, data)
    return tensor

def prob2logit(output):
  epsilon = 1e-7
  output = torch.clamp(output, epsilon, 1 - epsilon)
  output = torch.log(output / (1 - output))
  return output

def length(x):
  return torch.sum((x != 0).int(), 1)
