folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=sm.Unet \
  --backbone=resnet18 \
  --mix_dataset \
  --input='../input/quarter/tfrecords/train/*/*,../input/tfrecords/train/*/*' \
  --valid_exclude='../input/tfrecords' \
  --dataset_loss \
  --mname=$x \
  $*
