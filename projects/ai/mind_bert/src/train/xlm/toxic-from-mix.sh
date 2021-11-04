folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}
RECORD_DIR=../input/tfrecords/xlm
sh ./train/$v/common.sh \
    --model=xlm_model \
    --ckpt_dir=../working/exps/xlm/toxic-mix-unintended \
    --train_input=${RECORD_DIR}/jigsaw-toxic-comment-train \
    --valid_input=${RECORD_DIR}/validation \
    --pretrained=../input/tf-xlm-roberta-base \
    --gpus=8 \
    --vie=0.1 \
    --model_dir=../working/exps/$v/$x \
    $*
