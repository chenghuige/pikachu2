import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from lichee.module.torch.layer import brick as vn_layer
from lichee.module.torch.layer.det_conv_module import ConvModule
from lichee import plugin
from lichee.representation import representation_base


@plugin.register_plugin(plugin.PluginType.REPRESENTATION, "yolo_darkneck")
class YoloDarkNeckRepresentation(representation_base.BaseRepresentation):
    def __init__(self, representation_cfg):
        super(YoloDarkNeckRepresentation, self).__init__(representation_cfg)
        layer_list = [
            OrderedDict([
                ('head_body0_0', vn_layer.MakeNConv([512, 1024], 1024, 3)),
                ('spp', vn_layer.SpatialPyramidPooling()),
                ('head_body0_1', vn_layer.MakeNConv([512, 1024], 2048, 3)), ]
            ),
            OrderedDict([
                ('trans_0', vn_layer.FuseStage(512)),
                ('head_body1_0', vn_layer.MakeNConv([256, 512], 512, 5))]
            ),

            OrderedDict([
                ('trans_1', vn_layer.FuseStage(256)),
                ('head_body2_1', vn_layer.MakeNConv([128, 256], 256, 5))
            ]),  # out0

            OrderedDict([
                ('trans_2', vn_layer.FuseStage(128, is_reversal=True)),
                ('head_body1_1', vn_layer.MakeNConv([256, 512], 512, 5))]
            ),  # out1

            OrderedDict([
                ('trans_3', vn_layer.FuseStage(256, is_reversal=True)),
                ('head_body0_2', vn_layer.MakeNConv([512, 1024], 1024, 5))]
            ),  # out2
        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        out3, out4, out5 = x
        out5 = self.layers[0](out5)
        out4 = self.layers[1]([out4, out5])

        out3 = self.layers[2]([out3, out4])  # out0 large
        out4 = self.layers[3]([out3, out4])  # out1
        out5 = self.layers[4]([out4, out5])  # out2 small

        return [out5, out4, out3]
    
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
        new_state_dict = OrderedDict()
        for k, v in model_weight.items():
            # removing pooler layer weight
            if 'target.' in k:
                continue
            if 'pooler.dense' in k:
                continue

            k = re.sub('^module.', '', k)
            new_state_dict[k] = v
        return new_state_dict
