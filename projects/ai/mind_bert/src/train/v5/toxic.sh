folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}
RECORD_DIR=../input/tfrecords/xlm-192
sh ./train/$v/common.sh \
    --train_input=${RECORD_DIR}/jigsaw-toxic-comment-train \
    --pretrained=../input/tf-xlm-roberta-base \
    --model_dir=../working/exps/$v/$x \
    $*
