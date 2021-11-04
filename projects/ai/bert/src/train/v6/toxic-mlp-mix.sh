folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

bin=./train.py
# --train_input=../input/tfrecords/xlm/jigsaw-unintended-bias-train \
#    --train_input=../input/tfrecords/xlm/jigsaw-toxic-comment-train \
sh ./train/$v/toxic.sh \
    --train_input=../input/tfrecords/xlm/jigsaw-toxic-comment-train-google-tr-cleaned,../input/tfrecords/xlm/jigsaw-toxic-comment-train-google-it-cleaned,../input/tfrecords/xlm/jigsaw-toxic-comment-train-google-es-cleaned,../input/tfrecords/xlm/jigsaw-toxic-comment-train-google-ru-cleaned,../input/tfrecords/xlm/jigsaw-toxic-comment-train-google-pt-cleaned,../input/tfrecords/xlm/jigsaw-toxic-comment-train-google-fr-cleaned,../input/tfrecords/xlm/jigsaw-unintended-bias-train \
    --model=XlmModel \
    --use_mlp=1 \
    --vie=0.1 \
    --gpus=8 \
    --mname=$x \
    $*
