folder=$(dirname "$0")

x=$(basename "$0")
echo $x

bin=./train.py
echo $bin

$bin \
  --model="Model" \
  --learning_rate=0.001 \
  --min_learning_rate=1e-06 \
  --optimizer='bert-lazyadam' \
  --batch_size=256 \
  --eval_batch_size=512 \
  --reset_global_step=1 \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --vie=0.5 \
  --num_gpus=1 \
  $*

