folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x

bin=./run.py
echo $bin

$bin \
  --wandb_project='qqbrowser' \
  --model="baseline" \
  --model_dir="../working/$v/$x" \
  --static_input=1 \
  --input="../input/pointwise/*.tfrecords" \
  --valid_input="../input/pairwise/*.tfrecords" \
  --test_input="../input/test/*.tfrecords" \
  --batch_size=256 \
  --eval_batch_size=2048 \
  --learning_rate=0.0005 \
  --min_learning_rate=1e-06 \
  --lr_decay_power=1. \
  --batch_size_per_gpu=0 \
  --lr_scale=0 \
  --num_gpus=-1 \
  --optimizer='bert-adamw' \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --write_valid_final \
  --save_checkpoint=0 \
  --first_interval_epoch=0.1 \
  --vie=1 \
  --cache_valid=0 \
  --fp16=0 \
  --async_valid=0 \
  --async_eval=0 \
  --eval_leave=1 \
  --print_depth=1 \
  --seed=1024 \
  --epochs=8 \
  $*


