folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x

bin=./run.py
echo $bin

$bin \
  --wandb_project='qqbrowser' \
  --wandb=0 \
  --model_dir="../input/pretrain/word/$v/$x" \
  --word \
  --custom_model \
  --vocab_size=100000 \
  --use_vision \
  --epochs=10 \
  --batch_size=256 \
  --learning_rate=3e-4 \
  --min_learning_rate=1e-06 \
  --num_gpus=-1 \
  --optimizer='bert-lazyadam' \
  --interval_steps=100 \
  --save_checkpoint=0 \
  --save_interval_epochs=1 \
  --use_vision \
  --rv=0 \
  --fp16=0 \
  $*

