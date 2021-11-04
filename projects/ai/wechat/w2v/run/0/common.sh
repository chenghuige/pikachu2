folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x

bin=./main.py
echo $bin

$bin \
  --wandb_project='wechat_w2v' \
  --model_dir="../working/w2v/$v/$x" \
  --epochs=10 \
  --sample_method=log_uniform \
  --records_name=tfrecords/w2v \
  --batch_size=4096 \
  --eval_batch_size=4096 \
  --learning_rate=0.001 \
  --min_learning_rate=1e-06 \
  --lr_decay_power=0.5 \
  --batch_size_per_gpu=0 \
  --lr_scale=0 \
  --num_gpus=2 \
  --optimizer='bert-lazyadam' \
  --interval_steps=100 \
  --save_checkpoint=0 \
  --save_interval_epochs=1 \
  --fp16=0 \
  $*

