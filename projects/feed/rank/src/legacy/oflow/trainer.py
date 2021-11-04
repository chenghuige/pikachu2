import os
import time
import numpy as np
from datetime import datetime

import oneflow as flow

import config as configs
import model as wide_n_deep
import util as util

from sklearn.metrics import roc_auc_score


def ParameterUpdateStrategy():
    return {
        #'learning_rate_decay': {
        #     'polynomial_conf' : {
        #     'decay_batches': 1000,
        #     'end_learning_rate': 0.0
        #      }
        # },
        #'warmup_conf': {
        #     'linear_conf': {
        #          'warmup_batches': 0,
        #          'start_multiplier': 0
        #       }
        # },
        'lazy_adam_conf': {
        }
        #'adam_conf': {
        #    'epsilon': 1e-8,
        #}
    }


parser = configs.get_parser()
args = parser.parse_known_args()[0]

if args.enable_model_split:
    model_distribute = flow.distribute.split(axis=0)
else:
    model_distribute = flow.distribute.broadcast()


@flow.function
def Loss():
    total_device_num = args.node_num * args.device_num
    batch_size = total_device_num * args.batch_size_per_device

    #flow.config.default_initializer_conf(dict(constant_conf=dict(value=0.0)))
    flow.config.train.primary_lr(args.primary_lr)
    flow.config.train.model_update_conf(ParameterUpdateStrategy())

    decoder = util.Decoder(args.train_data_dir,
                           batch_size=batch_size, data_part_num=args.data_part_num)

    features = {}
    keys = ['feat_ids', 'feat_fields', 'feat_values', 'feat_masks']
    for i, key in enumerate(keys):
        features[key] = decoder[i]

    labels = decoder[4]
    loss, pred, ref = wide_n_deep.WideAndDeep(args, features, labels,
                                              model_distribute=model_distribute)
    print(loss, pred, ref)
    flow.losses.add_loss(loss)
    return loss, pred, ref


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
        nodes = [{'addr':'192.168.1.15'},{'addr':'192.168.1.14'}]
        flow.config.machine(nodes[:args.node_num])

    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    snapshots_root_path = os.path.join(args.model_save_path,
                datetime.now().time().strftime('%H.%M.%S.%f'))
    snapshots_root_path = args.model_save_path
    # os.mkdir(snapshots_root_path)
    os.system('mkdir -p %s' % snapshots_root_path)
    check_point = flow.train.CheckPoint()
    if args.pretrain_model_path != '':
        assert os.path.isdir(args.pretrain_model_path)
        check_point.load(args.pretrain_model_path)
        print('init model from {}'.format(args.pretrain_model_path))
    else:
        check_point.init()
        print('init model on demand')

    fmt_str = "{:>12}  {:>12.10f}  {:>12.10f}  {:>12.10f}"
    print('{:>12}  {:14}  {}  {}'.format( "step", "loss", "auc", "time"))
    cur_time = time.time()
    for step in range(args.max_steps):
       loss, pred, ref = Loss().get()
       if step % args.loss_print_steps == 0:
           label_ = np.array(ref, dtype='float32')
           #auc = 0.0
           auc = roc_auc_score(label_, pred)
           print(fmt_str.format(step, loss.mean(), auc, time.time() - cur_time))
           cur_time = time.time()


       #if args.model_save_path != '':
       #    assert args.save_snapshot_after_batch_size > 0
       #    if step % args.save_snapshot_after_batch_size == 0:
       #        snapshot_save_path = os.path.join(snapshots_root_path, 'snapshot_%d'%(step+1))
       #        check_point.save(snapshot_save_path)
    snapshot_save_path = os.path.join(snapshots_root_path, 'snapshot')
    check_point.save(snapshot_save_path)


