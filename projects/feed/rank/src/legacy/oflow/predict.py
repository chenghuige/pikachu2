import os
import time
import numpy as np
from datetime import datetime

import oneflow as flow

import config as configs
import model as wide_n_deep
import util as util

from sklearn.metrics import roc_auc_score


parser = configs.get_parser()
args = parser.parse_known_args()[0]


@flow.function
def predict_job():
    total_device_num = args.node_num * args.device_num
    batch_size = total_device_num * args.batch_size_per_device

    decoder = util.Decoder(args.predict_data_dir, batch_size=batch_size,
                           data_part_num=args.predict_data_part_num)

    features = {}
    keys = ['feat_ids', 'feat_fields', 'feat_values', 'feat_masks']
    for i, key in enumerate(keys):
        features[key] = decoder[i]

    labels = decoder[4]
    return wide_n_deep.WideAndDeep(args, features, labels)


if __name__ == '__main__':

    for arg in vars(args):
        print('{} = {}'.format(arg, getattr(args, arg)))
    flow.config.gpu_device_num(args.device_num)
    flow.config.ctrl_port(9978)
    flow.config.data_port(9979)
    flow.config.default_data_type(flow.float)
    flow.config.enable_inplace(True)

    #assert args.node_num <= len(args.nodes)
    if args.node_num > 1:
        nodes = [{'addr':'192.168.1.15'},{'addr':'192.168.1.16'}]
        flow.config.machine(nodes[:args.node_num])
        flow.deprecated.init_worker(scp_binary=True, use_uuid=True)

    check_point = flow.train.CheckPoint()
    assert os.path.isdir(args.predict_model_path)
    check_point.load(args.predict_model_path)
    print('init model from {}'.format(args.predict_model_path))


    fmt_str = "{:>12}  {:>12.10f}  {:>12.10f}  {:>12.10f}"
    print('{:>12}  {:14}  {}  {}'.format( "step", "loss", "auc", "time"))
    labels = np.array([[0]])
    preds = np.array([[0]])
    cur_time = time.time()
    for step in range(args.max_predict_steps):
       loss, pred, ref = predict_job().get()
       label_ = np.array(ref, dtype='float32')
       labels = np.concatenate((labels, label_), axis=0)
       preds = np.concatenate((preds, pred), axis=0)
       auc = roc_auc_score(label_, pred)
       print(fmt_str.format(step, loss.mean(), auc, time.time() - cur_time))
       cur_time = time.time()
    print(labels.shape)
    auc = roc_auc_score(labels[1:], preds[1:])
    print(auc)
    if args.node_num > 1:
        flow.deprecated.delete_worker()

