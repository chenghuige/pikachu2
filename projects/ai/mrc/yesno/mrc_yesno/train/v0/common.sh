folder=$(dirname "$0")
v=${folder##*/}

x=$(basename "$0")
echo $x

bin=./train.py
echo $bin

$bin \
  --model="Model" \
  --model_dir="../working/$v/$x" \
  --train_input='../input/tfrecords-padded/train' \
  --valid_input='../input/tfrecords-padded/dev' \
  --model='Model' \
  --learning_rate=5e-5 \
  --min_learning_rate=1e-08 \
  --optimizer='bert-adamw' \
  --batch_size=8 \
  --fp16 \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --save_interval_steps=100000000000 \
  --write_summary=0 \
  --write_metric_summary=0 \
  --write_valid \
  --vie=1 \
  --allow_cpu=0 \
  --async_valid=0 \
  --write_valid \
  $*

