'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
LastEditTime: 2019-08-26 18:17:40
'''
from importer import *
import model.net as Net

def load_GPUS(model,model_path,kwargs):
    state_dict = torch.load(model_path,**kwargs)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model


def weight_init(m):
# 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0,0.001)
    elif isinstance(m, nn.BatchNorm2d):
        # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def build_model(cfg):
    '''
    Description: build_model
    Args (type): cfg(yaml)
    Return: model
    '''
    MODEL = cfg.MODEL
    if hasattr(Net,MODEL.NET_NAME):
        model = getattr(Net,MODEL.NET_NAME)()
        
    else:
        raise ValueError("No model found !")
    # model.apply(weight_init)
    if MODEL.LOAD_PATH != "":
        if torch.cuda.is_available():
            
            # state_dict = torch.load(MODEL.LOAD_PATH)
            state_dict = torch.load(MODEL.LOAD_PATH,map_location='cpu')
            model.load_state_dict(state_dict)
            # kwargs = {'map_location':lambda storage,loc:storage.cuda(0)}
            # model = load_GPUS(model,MODEL.LOAD_PATH,kwargs)
        else:
            state_dict = torch.load(MODEL.LOAD_PATH,map_location='cpu')
            model.load_state_dict(state_dict)
    return model