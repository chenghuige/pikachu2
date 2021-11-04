from lichee import plugin
from lichee.representation import representation_base


@plugin.register_plugin(plugin.PluginType.REPRESENTATION, "pass_through")
class PassThroughRepresentation(representation_base.BaseRepresentation):
    """Pass through repr layer provides

    Attributes
    ----------
    features: torch.nn.Sequential
        feature layers of vgg model.
    avgpool: torch.nn.AdaptiveAvgPool2d
        avg pool layer of vgg model.

    """
    def __init__(self, representation_cfg):
        super(PassThroughRepresentation, self).__init__(representation_cfg)

    def forward(self, *inputs):
        return inputs
