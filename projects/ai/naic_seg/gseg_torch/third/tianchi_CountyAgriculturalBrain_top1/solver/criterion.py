'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
LastEditTime: 2019-08-27 11:37:10
'''
from __future__ import print_function, division
from collections import OrderedDict

from importer import *
        


class cross_entropy2d(nn.Module):
    def __init__(self):
        super(cross_entropy2d,self).__init__()
    def forward(self,input, target, weight=[10/9,10/9,10/9,10/9,5/9], size_average=True):
        weight = torch.tensor(weight,device=target.device)
        # print(input.shape,target.shape)
        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = input.size()
        # log_p: (n, c, h, w)
        log_p = nn.functional.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = nn.functional.nll_loss(log_p, target.long(), weight=weight, reduction='sum')
        if size_average:
            loss /= mask.data.sum()
        return loss

class cross_entropy2d_drop_edge(nn.Module):
    def __init__(self,edge_size=32):
        super(cross_entropy2d_drop_edge,self).__init__()
        self.edge_size=edge_size
    def forward(self,input,target,weight=None,size_average=True):
        # weight = torch.tensor(weight,device=target.device)

        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = input.size()
        input = input[:,:,self.edge_size:h-self.edge_size,self.edge_size:w-self.edge_size]
        target = target[:,:,self.edge_size:h-self.edge_size,self.edge_size:w-self.edge_size]
        n, c, h, w = input.size()
        # log_p: (n, c, h, w)
        log_p = nn.functional.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = nn.functional.nll_loss(log_p, target.long(), weight=weight, reduction='sum')
        if size_average:
            loss /= mask.data.sum()
        return loss

class LabelSmoothing_drop_edge(nn.Module):
    def __init__(self,win_size=11,num_classes=5,smoothing=0.05,edge_size=32):
        super(LabelSmoothing_drop_edge, self).__init__()
        assert (win_size%2) == 1
        self.smoothing = smoothing /(num_classes-1)
        self.win_size = win_size
        self.num_classes = num_classes
        self.find_edge_Conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=win_size,padding=(win_size-1)//2,stride=1,bias=False)
        self.edge_size = edge_size
        
        new_state_dict = OrderedDict()
        weight = torch.zeros(1,1,win_size,win_size)
        weight = weight -1
        weight[:,:,win_size//2,win_size//2] = win_size*win_size - 1
        new_state_dict['weight'] = weight
        self.find_edge_Conv.load_state_dict(new_state_dict)
        self.find_edge_Conv.weight.requires_grad=False

    def to_categorical(self,y,alpha=0.05,num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical = categorical + alpha
        categorical[np.arange(n), y] = (1-alpha) + (alpha/self.num_classes)
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        categorical = categorical.transpose(0,3,1,2)
        return categorical
        

    def forward(self, x, target):
        assert x.size(1) == self.num_classes
        h,w = x.shape[-2],x.shape[-1]
        x = x[:,:,self.edge_size:h-self.edge_size,self.edge_size:w-self.edge_size]
        target = target[:,:,self.edge_size:h-self.edge_size,self.edge_size:w-self.edge_size]

        log_p = nn.functional.log_softmax(x,dim=1)
        self.find_edge_Conv.cuda(device=target.device)
        edge_mask = self.find_edge_Conv(target)
        edge_mask = edge_mask.data.cpu().numpy()
        edge_mask[edge_mask!=0] = 1
        self.smoothing = np.mean(edge_mask)
        if self.smoothing > 0.2:
            self.smoothing = 0.2
        
        target = target.squeeze(dim=1)
        target = target.data.cpu().numpy()
        onehot_mask = self.to_categorical(target,0,num_classes=self.num_classes)
        onehot_mask = onehot_mask*(1-edge_mask)
        softlabel_mask = self.to_categorical(target,alpha=self.smoothing,num_classes=self.num_classes)
        softlabel_mask = softlabel_mask*edge_mask
        onehot_mask = torch.from_numpy(onehot_mask).cuda(device=log_p.device).float()
        softlabel_mask = torch.from_numpy(softlabel_mask).cuda(device=log_p.device).float()
        loss = torch.sum(onehot_mask*log_p+softlabel_mask*log_p,dim=1).mean()
        return -loss

class LabelSmoothing_fixsmoothing_drop_edge(nn.Module):
    def __init__(self, size=5, smoothing=0.045,edge_size=32):
        super(LabelSmoothing_fixsmoothing_drop_edge, self).__init__()
        self.confidence = (1.0 - smoothing)#if i=y的公式
        self.smoothing = smoothing /(size-1)
        self.size = size
        self.edge_size = edge_size

    def to_categorical(self,y,alpha=0.05,num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical = categorical + alpha
        categorical[np.arange(n), y] = 1-alpha + (alpha/num_classes)
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        categorical = categorical.transpose(0,3,1,2)
        return categorical
        

    def forward(self, x, target):
        assert x.size(1) == self.size
        # drop edge
        # x : B x C X H X W
        # target : B x C X H X W
        h,w = x.shape[-2],x.shape[-1]
        x = x[:,:,self.edge_size:h-self.edge_size,self.edge_size:w-self.edge_size]
        target = target[:,:,self.edge_size:h-self.edge_size,self.edge_size:w-self.edge_size]
        log_p = nn.functional.log_softmax(x,dim=1)

        target = target.squeeze(dim=1)
        mask = self.to_categorical(target.data.cpu().numpy(),self.smoothing,num_classes=self.size)
        mask = torch.from_numpy(mask).cuda(device=log_p.device).float()
        loss = torch.sum(mask*log_p,dim=1).mean()
        return -loss


class LabelSmoothing(nn.Module):
    def __init__(self,win_size=11,num_classes=5,smoothing=0.05):
        super(LabelSmoothing, self).__init__()
        assert (win_size%2) == 1
        self.smoothing = smoothing /(num_classes-1)
        self.win_size = win_size
        self.num_classes = num_classes
        self.find_edge_Conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=win_size,padding=(win_size-1)//2,stride=1,bias=False)
        
        new_state_dict = OrderedDict()
        weight = torch.zeros(1,1,win_size,win_size)
        weight = weight -1
        weight[:,:,win_size//2,win_size//2] = win_size*win_size - 1
        new_state_dict['weight'] = weight
        self.find_edge_Conv.load_state_dict(new_state_dict)
        self.find_edge_Conv.weight.requires_grad=False

    def to_categorical(self,y,alpha=0.05,num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical = categorical + alpha
        categorical[np.arange(n), y] = (1-alpha) + (alpha/self.num_classes)
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        categorical = categorical.transpose(0,3,1,2)
        return categorical
        

    def forward(self, x, target):
        assert x.size(1) == self.num_classes
        log_p = nn.functional.log_softmax(x,dim=1)
        self.find_edge_Conv.cuda(device=target.device)
        edge_mask = self.find_edge_Conv(target)
        edge_mask = edge_mask.data.cpu().numpy()
        edge_mask[edge_mask!=0] = 1
        self.smoothing = np.mean(edge_mask)
        if self.smoothing > 0.2:
            self.smoothing = 0.2
        
        target = target.squeeze(dim=1)
        target = target.data.cpu().numpy()
        onehot_mask = self.to_categorical(target,0,num_classes=self.num_classes)
        onehot_mask = onehot_mask*(1-edge_mask)
        softlabel_mask = self.to_categorical(target,alpha=self.smoothing,num_classes=self.num_classes)
        softlabel_mask = softlabel_mask*edge_mask
        onehot_mask = torch.from_numpy(onehot_mask).cuda(device=log_p.device).float()
        softlabel_mask = torch.from_numpy(softlabel_mask).cuda(device=log_p.device).float()
        loss = torch.sum(onehot_mask*log_p+softlabel_mask*log_p,dim=1).mean()
        return -loss

class LabelSmoothing_fixsmoothing(nn.Module):
    def __init__(self, size=5, smoothing=0.045):
        super(LabelSmoothing_fixsmoothing, self).__init__()
        self.confidence = (1.0 - smoothing)#if i=y的公式
        self.smoothing = smoothing /(size-1)
        self.size = size
        self.true_dist = None

    def to_categorical(self,y,alpha=0.05,num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical = categorical + alpha
        categorical[np.arange(n), y] = 1-alpha + (alpha/num_classes)
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        categorical = categorical.transpose(0,3,1,2)
        return categorical
        

    def forward(self, x, target):
        assert x.size(1) == self.size
        log_p = nn.functional.log_softmax(x,dim=1)

        target = target.squeeze(dim=1)
        mask = self.to_categorical(target.data.cpu().numpy(),self.smoothing,num_classes=self.size)
        mask = torch.from_numpy(mask).cuda(device=log_p.device).float()
        loss = torch.sum(mask*log_p,dim=1).mean()
        return -loss




class GIoULoss(nn.Module):
   def __init__(self):
       super(GIoULoss,self).__init__()
   
   def forward(self,predict,target):

       b1_x1,b1_y1,b1_x2,b1_y2 = predict[:,0],predict[:,1],predict[:,2],predict[:,3]
       b2_x1,b2_y1,b2_x2,b2_y2 = target[:,0],target[:,1],target[:,2],target[:,3]

       # 异常处理： 保证（x2,y2）>(x1,y1)
       b1_x1 = torch.min(b1_x1,b1_x2)
       b1_x2 = torch.max(b1_x1,b1_x2)
       b1_y1 = torch.min(b1_y1,b1_y2)
       b1_y2 = torch.max(b1_y1,b1_y2)
       
       b2_x1 = torch.min(b2_x1,b2_x2)
       b2_x2 = torch.max(b2_x1,b2_x2)
       b2_y1 = torch.min(b2_y1,b2_y2)
       b2_y2 = torch.max(b2_y1,b2_y2)

       # 求IOU
       inter_rect_x1 = torch.max(b1_x1,b2_x1)
       inter_rect_y1 = torch.max(b1_y1,b2_y1)
       inter_rect_x2 = torch.min(b1_x2,b2_x2)
       inter_rect_y2 = torch.min(b1_y2,b2_y2)
       # torch.clamp 将数值限制在大于等于0（处理极端情况IOU不相交）
       inter_area = torch.clamp(inter_rect_x2-inter_rect_x1,min=0)*torch.clamp(inter_rect_y2-inter_rect_y1,min=0)

       # 求 predict、target 的 smallest enclosing box
       outer_rect_x1 = torch.min(b1_x1,b2_x1)
       outer_rect_y1 = torch.min(b1_y1,b2_y1)
       outer_rect_x2 = torch.max(b1_x2,b2_x2)
       outer_rect_y2 = torch.max(b1_y2,b2_y2)
       outer_area = torch.clamp(outer_rect_x2-outer_rect_x1,min=0)*torch.clamp(outer_rect_y2-outer_rect_y1,min=0)
       
       target_box_area = torch.clamp(b2_x2-b2_x1,min=0)*torch.clamp(b2_y2-b2_y1,min=0)
       predict_box_area = torch.clamp(b1_x2-b1_x1,min=0)*torch.clamp(b1_y2-b1_y1,min=0)

       # predict_Box ∪ target_Box
       U = target_box_area + predict_box_area - inter_area

       GIoU = inter_area/(U+1e-10) - (outer_area-U)/(outer_area+1e-10)
       GIoU = torch.mean(GIoU)
       return 1 - GIoU

class GIoU_L1Loss(nn.Module):
    def __init__(self):
        super(GIoU_L1Loss,self).__init__()
        self.GIoULoss = GIoULoss()
        self.L1Loss = nn.L1Loss()
    
    def __call__(self,predict,target):
        loss1 = self.GIoULoss(predict,target)
        loss2 = self.L1Loss(predict,target)*10
        return loss1 + loss2
    
class GIoU_MSELoss(nn.Module):
    def __init__(self):
        super(GIoU_MSELoss,self).__init__()
        self.GIoULoss = GIoULoss()
        self.MSELoss = nn.MSELoss()
    
    def __call__(self,predict,target):
        loss1 = self.GIoULoss(predict,target)
        loss2 = self.MSELoss(predict,target)*10
        return loss1 + loss2

