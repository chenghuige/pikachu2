folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x

bin=./run.py
echo $bin

$bin \
  --wandb=1 \
  --wandb_tb=0 \
  --wandb_image=0 \
  --wandb_project='qqbrowser' \
  --write_summary=1 \
  --write_metrics_summary=1 \
  --write_valid_after_eval \
  --model="baseline" \
  --model_dir="../working/offline/$v/$x" \
  --static_input=1 \
  --input="../input/tfrecords/train/*.tfrec" \
  --valid_input="../input/tfrecords/valid/*.tfrec" \
  --test_input="../input/tfrecords/test/*.tfrec" \
  --parse_strategy=2 \
  --batch_size=512 \
  --eval_mul=4 \
  --learning_rate=0.0005 \
  --min_learning_rate=1e-07 \
  --lr_decay_power=1. \
  --batch_size_per_gpu=0 \
  --lr_scale=1. \
  --num_gpus=2 \
  --optimizer='bert-adamw' \
  --interval_steps=100 \
  --valid_interval_steps=0 \
  --write_valid_final \
  --save_checkpoint=0 \
  --save_graph=0 \
  --first_interval_epoch=-1 \
  --nvs=5 \
  --cache_valid=0 \
  --fp16=0 \
  --async_valid=0 \
  --async_eval=1 \
  --eval_leave=0 \
  --print_depth=1 \
  $*


