folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

bin=./train.py
# --train_input=../input/tfrecords/xlm/jigsaw-unintended-bias-train \
#    --train_input=../input/tfrecords/xlm/jigsaw-toxic-comment-train \
RECORD_DIR=../input/tfrecords/xlm
$bin \
    --model=XlmModel \
    --train_input=${RECORD_DIR}/jigsaw-toxic-comment-train \
    --pretrained=../input/tf-xlm-roberta-base \
    --valid_input=${RECORD_DIR}/validation \
    --test_input=${RECORD_DIR}/test \
    --interval_steps=100 \
    --valid_interval_steps=100 \
    --verbose=1 \
    --num_epochs=1 \
    --keras=1 \
    --buffer_size=2048 \
    --learning_rate=3e-5 \
    --optimizer=bert-adamw \
    --metrics='' \
    --test_names=id,toxic \
    --valid_interval_epochs=0.25 \
    --test_interval_epochs=-1 \
    --num_gpus=6 \
    --max_len=192 \
    --model_dir=../working/exps/$v/$x \
    $*
