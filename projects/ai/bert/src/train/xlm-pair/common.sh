folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

bin=./train.py
RECORD_DIR=../input/tfrecords/xlm-pair
$bin \
    --model=xlm_model \
    --train_input=${RECORD_DIR}/validation \
    --pretrained=../input/tf-xlm-roberta-base \
    --valid_input=${RECORD_DIR}/validation \
    --test_input=${RECORD_DIR}/test \
    --interval_steps=100 \
    --valid_interval_steps=0 \
    --verbose=1 \
    --write_metric_summary=1 \
    --write_summary=1 \
    --num_epochs=1 \
    --keras=1 \
    --buffer_size=2048 \
    --learning_rate=3e-5 \
    --opt_epsilon=1e-8 \
    --optimizer=bert-adamw \
    --metrics='acc,auc' \
    --test_names=id,toxic \
    --valid_interval_epochs=0.1 \
    --do_test=0 \
    --num_gpus=1 \
    --sparse_to_dense=1 \
    --padding_idx=1 \
    --batch_size=32 \
    --max_len=192 \
    --min_learning_rate=0 \
    --vie=0.5 \
    --model_dir=../working/exps/$v/$x \
    $*
