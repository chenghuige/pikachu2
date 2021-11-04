folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x

bin=./main.py
echo $bin

$bin \
  --wandb_project='wechat_rec' \
  --model="Model" \
  --model_dir="../working/offline/$v/$x" \
  --static_input=1 \
  --records_name=tfrecords \
  --records_name2=tfrecords \
  --batch_size=4096 \
  --eval_batch_size=8192 \
  --learning_rate=0.001 \
  --min_learning_rate=1e-06 \
  --lr_decay_power=0.5 \
  --batch_size_per_gpu=0 \
  --lr_scale=0 \
  --num_gpus=2 \
  --optimizer='bert-lazyadam' \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --write_valid_final \
  --save_checkpoint=0 \
  --first_interval_epoch=0 \
  --vie=0.25 \
  --fp16=0 \
  --async_valid=0 \
  --async_eval=1 \
  --eval_ab_users=1 \
  --always_eval_all=1 \
  --simple_eval=0 \
  --eval_leave=0 \
  --print_depth=-1 \
  --seed=1024 \
  $*


