import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable

class LargeEmbedding(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, num_devices=None, use_cuda=True, device_list=None, **kwargs):
    super(LargeEmbedding, self).__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    num_devices = num_devices or torch.cuda.device_count() 
    if not num_devices:
      num_devices = 1
      use_cuda = False
    self.num_devices = num_devices
    self.use_cuda = use_cuda


    self.num_pages = num_devices
    reminder = num_embeddings % num_devices
    if reminder:
      self.page_size = int((num_embeddings + num_devices - reminder) / num_devices)
    else:
      self.page_size = int(num_embeddings / num_devices)
    

    self.page_devices = []
    embedding_list = [nn.Embedding(self.page_size, embedding_dim, **kwargs) for i in range(self.num_pages)]
    if self.use_cuda:
      for i, embedding in enumerate(embedding_list):
        if device_list is not None:
          device = device_list[i]
          if (type(device) is list) or (type(device) is tuple):
            embedding_list[i] = nn.DataParallel(embedding, device_ids=device).cuda()
            self.page_devices.append(device[0])
          else:
            embedding.cuda(device)
            self.page_devices.append(device)
        else:
          device = i % self.num_devices
          embedding.cuda(device)
          self.page_devices.append(device)
    else:
        pass
    #
    self.embeddings = nn.ModuleList(embedding_list)

  def forward(self, indices_):
    indices = indices_.view(1, -1)
    y = torch.FloatTensor(1, indices.size(-1), self.embedding_dim)
    index_seq = torch.arange(0, indices.size(-1)).long().view(1, -1)
    if self.use_cuda:
      y = y.cuda()
      index_seq = index_seq.cuda()
      y = Variable(y)
      index_seq = Variable(index_seq, requires_grad=False)

      page_offset = 0
      for i in range(self.num_pages):
        mask_i = torch.min(torch.ge(indices, page_offset), torch.lt(indices, page_offset + self.page_size))
        #
        masked_idx_i = torch.masked_select(index_seq, mask_i)
        if masked_idx_i.dim() == 0:
          page_offset += self.page_size
          continue
        indices_i = torch.index_select(indices, 1, masked_idx_i) - page_offset
        if self.use_cuda:
          indices_i = indices_i.cuda(self.page_devices[i])
          try:
            v_i = self.embeddings[i](indices_i)
            v_i = v_i.cuda()
          except:
            print(indices_i, page_offset)
            print(self.page_devices[i])
            print(self.embeddings[i])
            print(self.embeddings[i].device_ids)
            print(indices_i.get_device())
            print(self.embeddings[i](indices_i.cuda(2)))
        else:
            v_i = self.embeddings[i](indices_i)
        y.index_copy_(1, masked_idx_i, v_i)
        #
        page_offset += self.page_size

    y = y.view(indices_.size(0), indices_.size(1), self.embedding_dim)

    return y

