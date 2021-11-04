folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=Model \
  --mdrop \
  --transformer=hfl/chinese-electra-180g-base-discriminator \
  --gpus=6 \
  --mname=$x \
  --train_input='../input/train_data_new/tfrecords/train_drop' \
  --valid_input='../input/tfrecords/dev' \
  $*
