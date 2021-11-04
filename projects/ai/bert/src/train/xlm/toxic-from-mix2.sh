folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}
RECORD_DIR=../input/tfrecords/xlm
sh ./train/$v/common.sh \
    --model=xlm_model \
    --pretrained=../working/exps/xlm/toxic-mix-unintended/tf-xlm-roberta-base \
    --train_input=${RECORD_DIR}/jigsaw-toxic-comment-train \
    --valid_input=${RECORD_DIR}/validation \
    --use_multi_dropout=1 \
    --gpus=8 \
    --vie=0.1 \
    --model_dir=../working/exps/$v/$x \
    $*
