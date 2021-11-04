import oneflow as flow

def _blob_conf(name, shape, dtype=flow.int32):
    return flow.data.BlobConf(name=name, shape=shape, dtype=dtype, codec=flow.data.RawCodec())

# def Decoder(data_dir='', batch_size=1, data_part_num=50, length=138):
#   #with flow.fixed_placement("cpu", "0:0"):
#     blob_confs = []
#     blob_confs.append(_blob_conf('index',  [length], flow.int64))
#     blob_confs.append(_blob_conf('field', [length], flow.int32))
#     blob_confs.append(_blob_conf('value', [length], flow.float))
#     blob_confs.append(_blob_conf('click',  [1]))
#     return flow.data.decode_ofrecord(data_dir, blob_confs, batch_size=batch_size, name="decode",
#                                      data_part_num=data_part_num, part_name_suffix_length=-1)

def decode(data_dir='', batch_size=1, data_part_num=50, length=138, shuffle=False):
  ofrecord = flow.data.ofrecord_reader(data_dir,
                                       batch_size=batch_size,
                                       data_part_num=data_part_num,
                                       part_name_suffix_length=-1,
                                       random_shuffle=shuffle,
                                       shuffle_after_epoch=shuffle)

  def _blob_decoder(bn, shape, dtype=flow.int32):
    return flow.data.OFRecordRawDecoder(ofrecord, bn, shape=shape, dtype=dtype)

  features = {
    'index': _blob_decoder('index', (length,), flow.int64),
    'field': _blob_decoder('field', (length,), flow.int32),
    'value': _blob_decoder('value', (length,), flow.float),
    'click': _blob_decoder('click', (1,)),
  }

  return features

