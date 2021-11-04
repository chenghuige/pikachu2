folder=$(dirname "$0")
v=${folder##*/}

x=$(basename "$0")
echo $x

bin=./train.py
echo $bin

$bin \
  --model="Model" \
  --model_dir="../working/$v/$x" \
  --train_input='../input/big/tfrecords/train' \
  --valid_input='../input/big/tfrecords/dev' \
  --test_input='../input/big/tfrecords/test' \
  --model='Model' \
  --restore_exclude=global_step,ignore,learning_rate \
  --global_epoch=0 \
  --global_step=0 \
  --learning_rate=0.01 \
  --min_learning_rate=1e-06 \
  --optimizer='bert-adam' \
  --batch_size=4096 \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --save_interval_steps=100000000000 \
  --write_summary \
  --write_metric_summary \
  --freeze_graph_final=0 \
  --vie=0.25 \
  --allow_cpu=0 \
  --async_valid \
  --max_history=100 \
  $*

