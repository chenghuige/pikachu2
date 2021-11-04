folder=$(dirname "$0")
v=${folder##*/}

x=$(basename "$0")
echo $x

bin=./train.py
echo $bin

#--train_input='../input/big/tfrecords/train-days/0' \
$bin \
  --model="Model" \
  --model_dir="../working/$v/$x" \
  --train_input='../input/big/tfrecords/train' \
  --valid_input='../input/big/tfrecords/dev' \
  --test_input='../input/big/tfrecords/test' \
  --model='Model' \
  --restore_exclude=global_step,ignore,learning_rate \
  --global_epoch=0 \
  --global_step=0 \
  --learning_rate=0.01 \
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
  --vie=0.25 \
  --allow_cpu=0 \
  --async_valid \
  --max_history=200 \
  --max_titles=50 \
  --shuffle=0 \
  --shuffle_batch=0 \
  --shuffle_files=0 \
  --num_valid=1000000 \
  --write_valid \
  --valid_mask_dids \
  --test_mask_dids \
  --train_mask_dids \
  --test_all_mask=0 \
  --mask_dids_ratio=-1 \
  $*

