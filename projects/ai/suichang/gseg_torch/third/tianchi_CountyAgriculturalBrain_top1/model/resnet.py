'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
@LastEditTime: 2019-11-02 16:49:34
'''
from importer import *

class ResNet_50(nn.Module):
  def __init__(self, in_channels = 3, conv1_out = 64):
    super(ResNet_50,self).__init__()
    backbone = models.resnet50(pretrained=True)
    # backbone = models.resnet50(pretrained=False)
    print("load pretrained ResNet50 model")
    # backbone.load_state_dict(torch.load(r"./model/resnet50-19c8e357.pth"))
    self.conv1 = backbone.conv1
    self.bn1 = backbone.bn1
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = backbone.maxpool
    self.layer1 = backbone.layer1
    self.layer2 = backbone.layer2
    self.layer3 = backbone.layer3
  
  def forward(self,x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    
    return x


class ResNet_101(nn.Module):
  def __init__(self, in_channels = 3, conv1_out = 64):
    super(ResNet_101,self).__init__()
    backbone = models.resnet101(pretrained=True)
    print("load pretrained ResNet101 model")
    # backbone.load_state_dict(torch.load(r"./model/resnet101-5d3b4d8f.pth"))
    self.conv1 = backbone.conv1
    self.bn1 = backbone.bn1
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = backbone.maxpool
    self.layer1 = backbone.layer1
    self.layer2 = backbone.layer2
    self.layer3 = backbone.layer3
  
  def forward(self,x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    
    return x


