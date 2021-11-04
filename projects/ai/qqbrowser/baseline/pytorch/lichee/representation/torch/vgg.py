from lichee import plugin
from lichee.representation import representation_base
import torch


vgg_params = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


@plugin.register_plugin(plugin.PluginType.REPRESENTATION, "vgg")
class VGGRepresentation(representation_base.BaseRepresentation):
    """VGG models are a type of CNN Architecture proposed by Karen Simonyan & Andrew Zisserman of
    Visual Geometry Group (VGG), Oxford University, which brought remarkable results for the ImageNet Challenge.
    Pre-trained model can be loaded. Details of VGG can be referred `here <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Attributes
    ----------
    features: torch.nn.Sequential
        feature layers of vgg model.
    avgpool: torch.nn.AdaptiveAvgPool2d
        avg pool layer of vgg model.

    """
    def __init__(self, representation_cfg):
        super(VGGRepresentation, self).__init__(representation_cfg)
        self.features = make_layers(vgg_params[representation_cfg["LAYERS"]],
                                    batch_norm=representation_cfg["BATCH_NORM"])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, inputs):
        """forward function of vgg model

        Parameters
        ----------
        inputs: torch.Tensor
            preprocessed image data by standard processes(ToSensor, Resize, Normalize)

        Returns
        ------
        x: torch.Tensor
            the output of vgg model representation layer, can be used in classifier
        """
        x = self.features(inputs)
        x = self.avgpool(x)
        return x

    @classmethod
    def load_pretrained_model(cls, cfg, pretrained_model_path):
        """load pre-trained model from specified path

        Parameters
        ----------
        cfg: list
            vgg params of each layer
        pretrained_model_path: Any
            path of pre-trained model

        Returns
        -------
        model: VGGRepresentation
            model loaded from pretrained_model_path

        """
        model = cls(cfg)
        state_dict = torch.load(pretrained_model_path, map_location="cpu")

        model.load_state_dict(state_dict, strict=True)
        return model


def make_layers(params, batch_norm=False):
    """Make the layers of VGG according to config.

    Parameters
    ----------
    params: list
        the params of each layer from global vgg_params
    batch_norm: bool
        if batch_norm is enabled

    Returns
    -------
    torch.nn.Sequential
        the sequential of the layer has been made
    """
    layers = []
    in_channels = 3
    for v in params:
        if v == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, torch.nn.ReLU(inplace=True)]
            in_channels = v
    return torch.nn.Sequential(*layers)
