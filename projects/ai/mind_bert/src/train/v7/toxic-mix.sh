folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}
RECORD_DIR=../input/tfrecords/xlm3
sh ./train/$v/common.sh \
    --train_input=${RECORD_DIR}/jigsaw-toxic-comment-train,../input/tfrecords/xlm/jigsaw-toxic-comment-train-google-tr-cleaned,../input/tfrecords/xlm/jigsaw-toxic-comment-train-google-it-cleaned,../input/tfrecords/xlm/jigsaw-toxic-comment-train-google-es-cleaned,../input/tfrecords/xlm/jigsaw-toxic-comment-train-google-ru-cleaned,../input/tfrecords/xlm/jigsaw-toxic-comment-train-google-pt-cleaned,../input/tfrecords/xlm/jigsaw-toxic-comment-train-google-fr-cleaned,../input/tfrecords/xlm/jigsaw-unintended-bias-train \
    --pretrained=../input/tf-xlm-roberta-base \
    --model_dir=../working/exps/$v/$x \
    $*
