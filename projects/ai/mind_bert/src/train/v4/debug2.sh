folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

bin=./train.py
# --train_input=../input/tfrecords/xlm/jigsaw-unintended-bias-train \
#    --train_input=../input/tfrecords/xlm/jigsaw-toxic-comment-train \
RECORD_DIR=../input/tfrecords/xlm2
sh ./train/$v/common.sh \
    --batch_sizes= \
    --buckets= \
    --batch_size=8 \
    --model_dir=../working/exps/$v/$x \
    $*
