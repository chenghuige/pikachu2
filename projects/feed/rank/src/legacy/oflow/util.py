import oneflow as flow

def _blob_conf(name, shape, dtype=flow.int32):
    return flow.data.BlobConf(name=name, shape=shape, dtype=dtype, codec=flow.data.RawCodec())

def Decoder(data_dir='', batch_size=1, data_part_num=1, length=400):
  #with flow.fixed_placement("cpu", "0:0"):
    blob_confs = []
    blob_confs.append(_blob_conf('feat_ids',    [length]))
    blob_confs.append(_blob_conf('feat_fields', [length]))
    blob_confs.append(_blob_conf('feat_values', [length], flow.float))
    blob_confs.append(_blob_conf('feat_masks',  [length], flow.float))
    blob_confs.append(_blob_conf('label',      [1]))
    return flow.data.decode_ofrecord(data_dir, blob_confs, batch_size=batch_size, name="decode",
                                     data_part_num=data_part_num, part_name_suffix_length=5)

