folder=$(dirname "$0")
v=${folder##*/}

x=$(basename "$0")
echo $x

bin=./train.py
echo $bin

export SM_FRAMEWORK=tf.keras
export PYTHONPATH=..:$PYTHONPATH

$bin \
  --model_dir="../working/$v/$x" \
  --train_input='../input/NWPU-RESISC45.tfrecords/*/*' \
  --restore_exclude=global_step,ignore,learning_rate \
  --global_epoch=0 \
  --global_step=0 \
  --learning_rate=1e-3 \
  --min_learning_rate=1e-5 \
  --lr_decay_power=1. \
  --optimizer='bert-adam' \
  --batch_size=32 \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --write_summary \
  --write_metric_summary \
  --freeze_graph_final=0 \
  --batch_parse=0 \
  --vie=1 \
  --epochs=5 \
  --allow_cpu=0 \
  --async_valid=0 \
  --print_depth=1 \
  --do_test=0 \
  --num_prefetch_batches=128 \
  --fold=0 \
  $*

