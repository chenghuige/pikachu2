folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x

bin=./run.py
echo $bin

$bin \
  --wandb_project='qqbrowser' \
  --wandb=1 \
  --model_dir="../working/pretrain/$v/$x" \
  --epochs=20 \
  --decay_epochs=0 \
  --batch_size=1024 \
  --learning_rate=5e-5 \
  --min_learning_rate=1e-06 \
  --num_gpus=-1 \
  --optimizer='bert-lazyadam' \
  --interval_steps=100 \
  --save_checkpoint=0 \
  --save_interval_epochs=1 \
  --fp16=0 \
  $*

