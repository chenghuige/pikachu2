import collections
import re

import torch
import torchvision

from lichee import plugin
from lichee.representation import representation_base


@plugin.register_plugin(plugin.PluginType.REPRESENTATION, "mobilenet_v2")
class MobileNetV2Representation(representation_base.BaseRepresentation):
    """MobileNetV2, that improves the state of the art performance of mobile models on multiple tasks and benchmarks
    as well as across a spectrum of different model sizes.
    Details of LongFormer can be referred `here <https://arxiv.org/abs/1801.04381>`_.
    """

    def __init__(self, representation_cfg):
        super(MobileNetV2Representation, self).__init__(representation_cfg)
        model = torchvision.models.mobilenet_v2(pretrained=False)
        self.features = model.features

    def forward(self, input):
        """forward function of vgg model

        Parameters
        ----------
        input: torch.Tensor
            preprocessed image data(ToSensor, Resize, Normalize)

        Returns
        ------
        output
            the output of mobilenet v2 model representation layer, can be used in classifier
        """
        output = self.features(input)
        return output

    @classmethod
    def load_pretrained_model(cls, representation_cfg, pretrained_model_path):
        model = cls(representation_cfg)

        state_dict = torch.load(pretrained_model_path,
                                map_location='cpu')

        state_dict = cls.state_dict_remove_pooler(state_dict)

        # Strict可以Debug参数
        model.load_state_dict(state_dict, strict=True)
        return model

    @classmethod
    def state_dict_remove_pooler(cls, model_weight):
        new_state_dict = collections.OrderedDict()
        for k, v in model_weight.items():
            # removing pooler layer weight
            if 'classifier.' in k:
                continue

            k = re.sub('^module.', '', k)
            new_state_dict[k] = v
        return new_state_dict
