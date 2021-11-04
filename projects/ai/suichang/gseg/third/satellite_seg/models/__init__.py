import torchvision.models as models

from fcn import *
from segnet import *
from pspnet import PSPNet

psp_models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

def get_instance(name):
    return {
            'fcn32s': fcn32s,
            'fcn8s': fcn8s,
            'fcn16s': fcn16s,
            'segnet': segnet,
        }[name]

def get_model(name, n_classes):
    if name in ['fcn32s', 'fcn16s', 'fcn8s']:
        fmodel = get_instance(name)
        model = fmodel(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == 'segnet':
        fmodel = get_instance(name)
        model = fmodel(n_classes=n_classes,
                      is_unpooling=True)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    elif name.find('pspnet-resnet152')!=-1:
        model = psp_models['resnet152']()
    elif name.find('pspnet-resnet101')!=-1:
        model = psp_models['resnet101']()
    elif name.find('pspnet-densenet')!=-1:
        model = psp_models['densenet']()
    else:
        raise 'Model {} not available'.format(name)

    return model
