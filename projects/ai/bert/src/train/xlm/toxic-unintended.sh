folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}
RECORD_DIR=../input/tfrecords/xlm
sh ./train/$v/common.sh \
    --ckpt_dir=../working/exps/unintended \
    --train_input=${RECORD_DIR}/jigsaw-toxic-comment-train \
    --pretrained=../input/tf-xlm-roberta-base \
    --gpus=6 \
    --model_dir=../working/exps/$v/$x \
    $*
