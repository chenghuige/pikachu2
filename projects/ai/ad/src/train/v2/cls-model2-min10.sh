folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --train_input="../input/tfrecords2/train" \
  --test_input="../input/tfrecords2/test" \
  --vocab_size=600000 \
  --model='ClsModel2' \
  --mname=$x \
  $*
