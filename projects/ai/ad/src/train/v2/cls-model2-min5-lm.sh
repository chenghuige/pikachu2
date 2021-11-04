folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --lm_target='ad_ids' \
  --train_input="../input/tfrecords/train" \
  --test_input="../input/tfrecords/test" \
  --vocab_size=1200000 \
  --model='ClsModel2' \
  --mname=$x \
  $*
