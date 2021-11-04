folder=$(dirname "$0")
v=${folder##*/}

x=$(basename "$0")
echo $x

bin=./train.py
echo $bin

$bin \
  --model="Model" \
  --model_dir="../working/$v/$x" \
  --train_input='../input/tfrecords/train-days' \
  --test_input='../input/tfrecords/test' \
  --valid_input='../input/tfrecords/dev' \
  --model='Model' \
  --restore_exclude=global_step,ignore,learning_rate \
  --global_epoch=0 \
  --global_step=0 \
  --learning_rate=0.001 \
  --min_learning_rate=1e-06 \
  --optimizer='bert-adam' \
  --batch_size=4096 \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --save_interval_steps=100000000000 \
  --write_summary \
  --write_metric_summary \
  --write_valid \
  --freeze_graph_final=0 \
  --start=0 \
  --loop_train \
  --loop_range \
  --loop_train_all \
  --loop_fixed_valid \
  --valid_span=-1 \
  --do_valid_last=False \
  --num_loop_dirs=7 \
  --rounds=0 \
  --vie=0.05 \
  --allow_cpu=0 \
  --async_valid \
  --max_history=200 \
  --max_titles=50 \
  --shuffle=0 \
  --num_valid=1000000 \
  --write_valid \
  --valid_mask_dids \
  --test_mask_dids \
  --train_mask_dids \
  --test_all_mask=0 \
  --mask_dids_ratio=-1 \
  $*

