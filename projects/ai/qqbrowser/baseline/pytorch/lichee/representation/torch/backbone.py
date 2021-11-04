import torch
from torch import nn
import torchvision

from lichee import plugin
from lichee.representation import representation_base


@plugin.register_plugin(plugin.PluginType.REPRESENTATION, "backbone")
class Backbone(representation_base.BaseRepresentation):
    """represents backbone network in video classification model

    Attributes
    ----------
    embedding: torch.nn.Module
        embedding network, can be resnet18, resnet34, resnet50, resnet101, resnet152 or inception_v3

    """

    def __init__(self, representation_cfg):
        super().__init__(representation_cfg)
        arch = representation_cfg['NAME']
        if arch in ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            model = getattr(torchvision.models, arch)(False)
        elif arch in ['Inception3', 'inception_v3']:
            model = getattr(torchvision.models, arch)(False)
            model.aux_logits = False
        else:
            raise ValueError('Unknown base model: {}'.format(arch))
        if not representation_cfg['PRETRAINED']:
            identity_layer = nn.Identity()
            identity_layer.in_features = model.classifier.in_features
            model.classifier = identity_layer
        self.embedding = model

    def forward(self, inputs):
        """forward function of backbone model

        Parameters
        ----------
        inputs: torch.Tensor
            input video frames, shoule be (batch, num_segments, channel, height, weight)

        Returns
        ------
        embedding: torch.Tensor
            output embedding feature
        """
        size = list(inputs.size())
        if len(size) == 5:
            """batch, num_segments, channel, height, weight"""
            inputs = inputs.view(-1, size[-3], size[-2], size[-1])
        embedding = self.embedding(inputs)
        if len(size) == 5:
            embedding_size = list(embedding.size())
            embedding = embedding.view((-1, size[1]) + tuple(embedding_size[1:]))
        return embedding

    @classmethod
    def load_pretrained_model(cls, cfg, pretrained_model_path):
        model = cls(cfg)
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        identity_layer = nn.Identity()
        identity_layer.in_features = model.fc.in_features
        model.fc = identity_layer
        return model
