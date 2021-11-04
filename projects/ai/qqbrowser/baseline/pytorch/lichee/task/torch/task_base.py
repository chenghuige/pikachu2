import torch.nn as nn

from lichee import config
from lichee import plugin


class BaseTask(nn.Module):
    """BaseTask class
    provide get_output method, use in prediction,
    TASK_OUT will get from configuration.
    Base class of Task, others task class should derived from this.
    """
    def __init__(self):
        super(BaseTask, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Not Implemented!")

    @classmethod
    def get_output(cls, logits):
        cfg = config.get_cfg()
        task_output_cls = plugin.get_plugin(plugin.PluginType.TASK_OUTPUT, cfg.MODEL.TASK.CONFIG.TASK_OUTPUT)
        return task_output_cls.get_output(logits)
