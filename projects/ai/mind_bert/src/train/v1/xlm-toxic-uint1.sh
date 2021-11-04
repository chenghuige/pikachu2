folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

bin=./train.py
# --train_input=../input/tfrecords/xlm/jigsaw-unintended-bias-train \
#    --train_input=../input/tfrecords/xlm/jigsaw-toxic-comment-train \
# --train_input=../input/tfrecords/xlm-sample1/jigsaw-unintended-bias-train,../input/tfrecords/xlm/jigsaw-toxic-comment-train \
$bin \
    --model=xlm_model \
    --train_input=../input/tfrecords/xlm/jigsaw-toxic-comment-train,../input/tfrecords/xlm-sample1/jigsaw-unintended-bias-train \
    --pretrained=../input/tf-xlm-roberta-base \
    --valid_input=../input/tfrecords/xlm/validation \
    --test_input=../input/tfrecords/xlm/test \
    --interval_steps=100 \
    --valid_interval_steps=100 \
    --verbose=1 \
    --num_epochs=1 \
    --keras=1 \
    --buffer_size=2048 \
    --learning_rate=3e-5 \
    --opt_epsilon=1e-8 \
    --optimizer=bert-adamw \
    --metrics='' \
    --test_names=id,toxic \
    --valid_interval_epochs=0.1 \
    --test_interval_epochs=0 \
    --batch_size=32 \
    --num_gpus=8 \
    --model_dir=../working/exps/$v/$x \
    $*
