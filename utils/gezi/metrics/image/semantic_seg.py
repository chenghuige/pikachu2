#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   sem_seg.py
#        \author   chenghuige  
#          \date   2020-09-27 20:46
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sklearn

import gezi
from gezi import logging, tqdm

# https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-3-17-09-55-segmentation/
class Evaluator(object):
    def __init__(self, num_classes, eval_each=False):
        self.class_names = []
        if isinstance(num_classes, (list, tuple)):
            self.class_names = num_classes
            self.num_classes = len(self.class_names)
        else:
            self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes,)*2)
        self.eval_each_ = eval_each
        self.iu = None
        self.inited = False

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def _calc_iu(self):
        return np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + 
            np.sum(self.confusion_matrix, axis=0)-
            np.diag(self.confusion_matrix))

    def Mean_Intersection_over_Union(self):
        # [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        iu =  self._calc_iu()
        MIoU = np.nanmean(iu)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = self._calc_iu()
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def iou(self):
        iu =  self._calc_iu()
        MIoU = np.nanmean(iu)
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return MIoU, FWIoU, iu

    def _generate_matrix(self, gt_image, pre_image):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        # confusion_matrix = sklearn.metrics.confusion_matrix(y_true=label.reshape(-1), y_pred=pre_image.reshape(-1))
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, f'{gt_image.shape} {pre_image.shape}'
        if not self.eval_each_:
            self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        else:
            for i in tqdm(range(len(gt_image)), 'confusion_matrixes'):
                self.confusion_matrixes += [self._generate_matrix(gt_image[i], pre_image[i])]
            self.confusion_matrix = np.sum(self.confusion_matrixes)
        self.inited = True
        return self

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)
        self.inited = False

    def eval_once(self, metric=None):
    #   with gezi.Timer('semantic seg eval', print_fn=logging.info):
        assert self.inited, 'call add_batch(label_image, pred) first'

        if metric is not None:
            if metric.lower() == 'fwiou':
                metric = 'Frequency_Weighted_Intersection_over_Union' 
            elif metric.lower() == 'miou' or metric.lower() == 'iou':
                metric = 'Mean_Intersection_over_Union'
            metric_fn = getattr(self, metric)
            return metric_fn()

        res = {}
        iou, fwiou, iu = self.iou()
        res['FWIoU'] = fwiou
        res['MIoU'] = iou
        res['ACC/pixel'] = self.Pixel_Accuracy()
        res['ACC/class'] = self.Pixel_Accuracy_Class()
        if self.class_names:
            for i in range(len(self.class_names)):
                class_name = self.class_names[i]
                res[f'IoU/{class_name}'] = iu[i]
        return res

    def eval_each(self, gt_image, pre_image, metric=None):
        assert gt_image.shape == pre_image.shape, f'{gt_image.shape} {pre_image.shape}'
        confusion_matrix = self.confusion_matrix
        res_all = {} if not metric else []
        self.inited = True
        # for i in tqdm(range(len(gt_image)),  ascii=True, desc='eval_each'):
        for i in range(len(gt_image)):
            self.confusion_matrix = self._generate_matrix(gt_image[i], pre_image[i])
            confusion_matrix += self.confusion_matrix
            res = self.eval_once(metric)
            if not metric:
                for key in res:
                    if key not in res_all:
                        res_all[key] = [res[key]]  
                    else:
                        res_all[key] += [res[key]]
            else:
                res_all += [res]
        self.confusion_matrix = confusion_matrix
        return res_all

    def eval(self, gt_image, pre_image, return_all=False):
        if not return_all:
            self.add_batch(gt_image, pre_image)
            return self.eval_once()
        else:
            return self.eval_all(gt_image, pre_image)

    def eval_all(self, gt_image, pre_image):
        res_all = {}
        confusion_matrix = np.zeros((self.num_classes,) * 2)
        self.inited = True
        for i in tqdm(range(len(gt_image)), ascii=True, desc='confusion_matrixes'):
            self.confusion_matrix = self._generate_matrix(gt_image[i], pre_image[i])
            confusion_matrix += self.confusion_matrix
            res = self.eval_once()
            for key in res:
                if key not in res_all:
                    res_all[key] = [res[key]]  
                else:
                    res_all[key] += [res[key]]
        self.confusion_matrix = confusion_matrix
        res = self.eval_once()
        return res, res_all

