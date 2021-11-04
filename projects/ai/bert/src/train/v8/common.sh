folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

bin=./train.py
# --train_input=../input/tfrecords/xlm/jigsaw-unintended-bias-train \
#    --train_input=../input/tfrecords/xlm/jigsaw-toxic-comment-train \
RECORD_DIR=../input/tfrecords/xlm-pair
$bin \
    --model=XlmModel \
    --train_input=${RECORD_DIR}/validation \
    --pretrained=../input/tf-xlm-roberta-base \
    --valid_input=${RECORD_DIR}/validation \
    --test_input=${RECORD_DIR}/test \
    --interval_steps=100 \
    --valid_interval_steps=0 \
    --verbose=1 \
    --num_epochs=1 \
    --keras=1 \
    --buffer_size=2048 \
    --learning_rate=3e-5 \
    --opt_epsilon=1e-8 \
    --optimizer=bert-adamw \
    --metrics='' \
    --test_names=id,toxic \
    --valid_interval_epochs=0.25 \
    --do_test=0 \
    --num_gpus=1 \
    --sparse_to_dense=1 \
    --padding_idx=1 \
    --buckets=190,350 \
    --batch_size=32 \
    --batch_sizes=32,16,8 \
    --min_learning_rate=0 \
    --length_key=input_word_ids \
    --model_dir=../working/exps/$v/$x \
    $*
