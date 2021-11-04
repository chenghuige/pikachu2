'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
LastEditTime: 2019-07-02 15:16:23
'''
import util.metric as Metric

def make_metrics(cfg):
    if hasattr(Metric,cfg.UTILS.METRICS):
        metrics = getattr(Metric,cfg.UTILS.METRICS)()
        return metrics
    else:
        raise Exception("Invalid metric",cfg.UTILS.METRICS)