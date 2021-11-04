folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}
RECORD_DIR=../input/tfrecords/xlm
sh ./train/$v/common.sh \
    --model=xlm_model \
    --ckpt_dir=../working/exps/unintended \
    --train_input=${RECORD_DIR}/jigsaw-toxic-comment-train,${RECORD_DIR}/jigsaw-toxic-comment-train-google-tr-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-it-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-es-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-ru-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-pt-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-fr-cleaned \
    --valid_input=${RECORD_DIR}/validation \
    --pretrained=../input/tf-xlm-roberta-base \
    --gpus=8 \
    --model_dir=../working/exps/$v/$x \
    $*
