folder=$(dirname "$0")
v=${folder##*/}

x=$(basename "$0")
echo $x

bin=./train.py
echo $bin

$bin \
  --model="Model" \
  --model_dir="../working/$v/$x" \
  --wandb_project='imt2020' \
  --train_input='../input/tfrecords/train/*' \
  --valid_input='../input/tfrecords/train/record_0.*' \
  --model='Model' \
  --learning_rate=1e-3 \
  --min_learning_rate=1e-08 \
  --optimizer='bert-adamw' \
  --batch_size=32 \
  --fp16=0 \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --save_interval_steps=100000000000 \
  --write_summary=0 \
  --write_metric_summary=0 \
  --epochs=20 \
  --num_eval_steps=10 \
  --allow_cpu=0 \
  --async_valid=0 \
  --write_valid=0 \
  --gpus=4 \
  $*

