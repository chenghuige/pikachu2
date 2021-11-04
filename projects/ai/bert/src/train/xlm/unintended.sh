folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}
RECORD_DIR=../input/tfrecords/xlm
sh ./train/$v/common.sh \
    --train_input=${RECORD_DIR}/jigsaw-unintended-bias-train \
    --pretrained=../input/tf-xlm-roberta-base \
    --vie=0.1 \
    --model_dir=../working/exps/$v/$x \
    $*
