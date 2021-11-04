import os
import time
import numpy as np
from datetime import datetime

import oneflow as flow

import config as configs
import model as wide_n_deep
import util as util

from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import gezi


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


@flow.global_function(type='train')
def Loss():
    total_device_num = args.node_num * args.device_num
    batch_size = total_device_num * args.batch_size_per_device

    iter_num = 1000
    warmup_proportion = 0.1
    warmup_batches = int(iter_num * warmup_proportion)
    lr_warmup = flow.optimizer.warmup.linear(warmup_batches, 0)
    lr_scheduler = flow.optimizer.PolynomialSchduler(args.primary_lr, iter_num, 0.0, 
                                                     warmup=lr_warmup)
    opt = flow.optimizer.LazyAdam(lr_scheduler, epsilon=1e-6, 
                                  grad_clipping=flow.optimizer.grad_clipping.by_global_norm(1.0))

    features = util.decode(args.train_data_dir, batch_size=batch_size)

    labels = features['click']

    loss, pred, ref = wide_n_deep.WideAndDeep(args, features, labels,
                                              model_distribute=model_distribute)
    opt.minimize(loss)
    return loss, pred, ref


if __name__ == '__main__':

    for arg in vars(args):
        print('{} = {}'.format(arg, getattr(args, arg)))
    flow.config.gpu_device_num(args.device_num)

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
    # for step in range(args.max_steps):
    for step in tqdm(range(args.max_steps)):
       loss, pred, label = Loss().get()
       if step % args.loss_print_steps == 0:
           label = gezi.squeeze(label.numpy())
           pred = gezi.squeeze(pred.numpy())
           auc = roc_auc_score(label, pred)
           print(step, auc)
        #    print(fmt_str.format(step, loss.numpy(), auc, time.time() - cur_time))
           cur_time = time.time()


       #if args.model_save_path != '':
       #    assert args.save_snapshot_after_batch_size > 0
       #    if step % args.save_snapshot_after_batch_size == 0:
       #        snapshot_save_path = os.path.join(snapshots_root_path, 'snapshot_%d'%(step+1))
       #        check_point.save(snapshot_save_path)
    snapshot_save_path = os.path.join(snapshots_root_path, 'snapshot')
    check_point.save(snapshot_save_path)


