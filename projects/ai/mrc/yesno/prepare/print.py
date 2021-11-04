#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   print.py
#        \author   chenghuige  
#          \date   2021-02-08 15:38:45.850594
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

def get_train_test(version='v7', if_use_title=True, if_label_discrete=False, ann_date='20210111',data_dir='../input'):
    postfix = '.model_in.obj'
    if if_use_title:
        postfix = '.use_title.model_in.obj'
    train_data, dev_data = [], []
    mark_dir = data_dir + '/mark/'
    mark_train_path = mark_dir + 'yesno_mark_1.3w.token_process.train' + postfix
    mark_test_2k_path = mark_dir + 'yesno_mark_1.3w.token_process.test_2000' + postfix

    # mark 2w
    mark_2w_dir = data_dir + '/mark_2w/'
    mark_2w_path = mark_2w_dir + 'yesno_mark_2w_change.token_process' + postfix

    # ann 10w
    ann_10w_dir = data_dir + '/ann_data/20210111/'
    ann_10w_path = ann_10w_dir + 'data.20210111.100000_change.token_process' + postfix

    # ann unk less
    unk_less_dir = data_dir + '/ann_data/unk_data/'
    unk_7w_path = unk_less_dir + 'unk.77462.token_process' + postfix
    unk_3k_path = unk_less_dir + 'unk.3130.token_process' + postfix

    # test_set
    test_dir = data_dir + '/test_set_data/'
    test_path = test_dir + 'test_set_cases_all_4_change_rep.token_process' + postfix

    # aic 5w 以及 train 10w dev 1w数据
    data_dir_ori = '/search/odin/huanghouwen/LizhiPretrainLM_1214/data/finetune/yesno_data/ori_data/'
    path_ori_list = [data_dir_ori + 'aic_cqa_new_processed._split_sent.token_process' + postfix,
                     data_dir_ori + 'train_data_0317_all._split_sent.token_process' + postfix,
                     data_dir_ori + 'dev_data_0317_all._split_sent.token_process' + postfix]

    if version == 'v7':
        train_path_list = [ mark_train_path, mark_test_2k_path, mark_2w_path,
                            ann_10w_path, unk_7w_path, unk_3k_path
                           ]
        dev_path_list = [test_path]


    print(','.join(train_path_list))
    print(dev_path_list)

get_train_test()
