folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}
RECORD_DIR=../input/tfrecords/xlm-pair
sh ./train/$v/common.sh \
    --model=xlm_model \
    --pretrained=../working/exps/xlm/toxic-mix-unintended/tf-xlm-roberta-base \
    --use_word_ids2=1 \
    --train_input=${RECORD_DIR}/jigsaw-toxic-comment-train,${RECORD_DIR}/jigsaw-toxic-comment-train-google-tr-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-it-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-es-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-ru-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-pt-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-fr-cleaned \
    --valid_input=${RECORD_DIR}/validation \
    --gpus=8 \
    --vie=0.1 \
    --model_dir=../working/exps/$v/$x \
    $*
