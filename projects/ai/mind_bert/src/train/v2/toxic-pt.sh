folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

bin=./train.py
# --train_input=../input/tfrecords/xlm/jigsaw-unintended-bias-train \
#    --train_input=../input/tfrecords/xlm/jigsaw-toxic-comment-train \
sh ./train/$v/toxic.sh \
    --train_input=../input/tfrecords/xlm/jigsaw-toxic-comment-train-google-pt \
    --mname=$x \
    $*
