folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}
RECORD_DIR=../input/tfrecords/xlm2
sh ./train/$v/common.sh \
    --train_input=${RECORD_DIR}/jigsaw-toxic-comment-train \
    --gpus=8 \
    --use_multi_dropout=1 \
    --dropout=0.3 \
    --vie=0.1 \
    --model_dir=../working/exps/$v/$x \
    $*
