folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --train_input=../input/tfrecords.v1/train \
  --test_input=../input/tfrecords.v1/eval \
  --learning_rate=0.001 \
  --min_count=5 \
  --use_w2v \
  --words_w2v \
  --pooling=dot \
  --model=Model \
  --mname=$x \
  $*
