folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}
RECORD_DIR=../input/tfrecords/xlm
sh ./train/$v/common.sh \
    --train_input=${RECORD_DIR}/jigsaw-toxic-comment-train \
    --pretrained=../input/tf-xlm-roberta-base \
    --use_multi_dropout=1 \
    --dropout=0.5 \
    --model_dir=../working/exps/$v/$x \
    $*
