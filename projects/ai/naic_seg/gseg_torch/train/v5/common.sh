folder=$(dirname "$0")
v=${folder##*/}

x=$(basename "$0")
echo $x

bin=./train.py
echo $bin

export SM_FRAMEWORK=tf.keras
export PYTHONPATH=..:$PYTHONPATH

$bin \
  --model_dir="../working/$v.torch/$x" \
  --train_input='../input/tfrecords/train/*/*' \
  --restore_exclude=global_step,ignore,learning_rate \
  --global_epoch=0 \
  --global_step=0 \
  --learning_rate=5e-4 \
  --min_learning_rate=1e-6 \
  --lr_decay_power=2. \
  --optimizer='bert-Adam' \
  --batch_size=32 \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --write_summary \
  --write_metric_summary \
  --freeze_graph_final=0 \
  --vie=1 \
  --epochs=1 \
  --torch_only=1 \
  --allow_cpu=0 \
  --async_valid=0 \
  --print_depth=1 \
  --do_test=0 \
  --cache_valid=0 \
  --num_epochs=50 \
  --num_prefetch_batches=12 \
  --fold=1 \
  --num_workers=6 \
  $*

