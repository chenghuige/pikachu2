folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x

bin=./train.py
echo $bin

$bin \
  --wandb_project='wechat_rec' \
  --model="Model" \
  --model_dir="../working/offline/$v/$x" \
  --learning_rate=0.001 \
  --min_learning_rate=1e-06 \
  --optimizer='lazyadam' \
  --batch_size=256 \
  --eval_batch_size=512 \
  --reset_global_step=1 \
  --interval_steps=100 \
  --valid_interval_steps=0 \
  --num_eval_steps=2 \
  --num_gpus=1 \
  $*

