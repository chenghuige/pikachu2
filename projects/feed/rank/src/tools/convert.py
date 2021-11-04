# -*- coding: utf-8 -*-
"""
Filename: convertor_wide_and_deep_concat.py
Description:
Copyright: Copyright (c) Sogou Inc.
Company: Sogou Inc.
Author: Huashen Liang
Version: 1.0
E-mail: lianghuashen@sogou-inc.com
Last modified: 2019-07-16 15:16
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import struct
import sys


def write_text(tensor_var, ofs, name):
    dim = len(tensor_var.shape)
    ofs.write(('%s: ' % name).encode())
    ofs.write(' '.join([str(d) for d in tensor_var.shape]))
    ofs.write('\n')
    if dim == 1:
        line_str_list = [str(d) for d in tensor_var]
        ofs.write(str(' '.join(line_str_list)))
    elif dim == 2:
        for i in range(tensor_var.shape[0]):
            line_str_list = [str(d) for d in tensor_var[i]]
            ofs.write(str(' '.join(line_str_list)))
            ofs.write('\n')


def write_matrix_bin(tensor_var, ofs, name):
    print('writing "%s"' % name)
    rows = tensor_var.shape[0]
    cols = tensor_var.shape[1]
    ofs.write(('%s: ' % name).encode())
    ofs.write(struct.pack("=i", rows))
    ofs.write(struct.pack("=i", cols))
    for r in range(tensor_var.shape[0]):
        for c in range(tensor_var.shape[1]):
            ofs.write(struct.pack("=f", tensor_var[r][c]))


def write_vector_bin(tensor_var, ofs, name):
    print('writing "%s"' % name)
    dim = tensor_var.shape[0]

    ofs.write(('%s: ' % name).encode())
    ofs.write(struct.pack("=i", dim))
    if len(tensor_var.shape) == 2:
        for i in range(dim):
            ofs.write(struct.pack("=f", tensor_var[i][0]))
    elif len(tensor_var.shape) == 1:
        for i in range(dim):
            ofs.write(struct.pack("=f", tensor_var[i]))
    else:
        sys.stderr.write('unknown tensor shape %d' % len(tensor_var.shape))


def write_scalar_bin(tensor_var, ofs, name):
    print('writing "%s"' % name)
    print('scaler="%f"' % tensor_var[0])
    ofs.write(('%s: ' % name).encode())
    ofs.write(struct.pack("=f", tensor_var[0]))


# wide_deep/deep/dense_1/bias (DT_FLOAT) [1]
# wide_deep/deep/dense_1/kernel (DT_FLOAT) [50,1]

# wide_deep/deep/emb/embeddings (DT_FLOAT) [2976660,50]
# wide_deep/deep/field_emb/embeddings (DT_FLOAT) [69,50]

# wide_deep/deep/mlp/dense/bias (DT_FLOAT) [50]
# wide_deep/deep/mlp/dense/kernel (DT_FLOAT) [100,50]

# wide_deep/dense_2/bias (DT_FLOAT) [1]
# wide_deep/dense_2/kernel (DT_FLOAT) [2,1]

# wide_deep/wide/bias (DT_FLOAT) [1]
# wide_deep/wide/emb/embeddings (DT_FLOAT) [2976661,1]

def main():
    tf_ckpt = sys.argv[1]
    path = sys.argv[2]
    ofs = open(path, 'wb')

    #------------------------------------------deprecated
    ofs.write('none'.encode())
    return

    reader = tf.compat.v1.train.NewCheckpointReader(tf_ckpt)

    #-------wide
    wide_emb = reader.get_tensor('wide_deep/wide/emb/embeddings')
    write_vector_bin(wide_emb, ofs, 'wide_emb')

    wide_bias = reader.get_tensor('wide_deep/wide/bias')
    write_scalar_bin(wide_bias, ofs, 'wide_bias')

    #--------deep
    deep_emb = reader.get_tensor('wide_deep/deep/emb/embeddings')
    write_matrix_bin(deep_emb, ofs, 'deep_emb')

    deep_field_emb = reader.get_tensor('wide_deep/deep/field_emb/embeddings')
    write_matrix_bin(deep_field_emb, ofs, 'deep_field_emb')

    deep_mlp_dense_kernel = reader.get_tensor('wide_deep/deep/mlp/dense/kernel')
    #write_vector_bin(deep_mlp_dense_kernel, ofs, 'deep_mlp_dense_kernel')
    write_matrix_bin(deep_mlp_dense_kernel, ofs, 'deep_mlp_dense_kernel')
    deep_mlp_dense_bias = reader.get_tensor('wide_deep/deep/mlp/dense/bias')
    #write_scalar_bin(deep_mlp_dense_bias, ofs, 'deep_mlp_dense_bias')
    write_vector_bin(deep_mlp_dense_bias, ofs, 'deep_mlp_dense_bias')

    deep_dense_kernel = reader.get_tensor('wide_deep/deep/dense_1/kernel')
    write_vector_bin(deep_dense_kernel, ofs, 'deep_dense_kernel')
    deep_dense_bias = reader.get_tensor('wide_deep/deep/dense_1/bias')
    write_scalar_bin(deep_dense_bias, ofs, 'deep_dense_bias')

    #--------w&d
    wd_dense_kernel = reader.get_tensor('wide_deep/dense_2/kernel')
    write_vector_bin(wd_dense_kernel, ofs, 'wd_dense_kernel')
    wd_dense_bias = reader.get_tensor('wide_deep/dense_2/bias')
    write_scalar_bin(wd_dense_bias, ofs, 'wd_dense_bias')


if __name__ == '__main__':
    main()
