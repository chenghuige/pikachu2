folder=$(dirname "$0")
v=${folder##*/}

x=$(basename "$0")
echo $x

bin=./train.py
#if [ $INFER ];then
#  if [[ $INFER == "1" ]]
#  then
#     bin=./infer.py
#  fi
#fi
echo $bin
# --restore_exclude=global_step,ignore,learning_rate,OptimizeLoss,Adam \
# --test_input="../input/tfrecords/eval" \
# --test_input="../input/tfrecords/train/30" \
$bin \
  --model="Baseline" \
  --model_dir="../working/$v/$x" \
  --train_input="../input/tfrecords/train/1" \
  --valid_input="../input/tfrecords/train/2" \
  --test_input="../input/tfrecords/eval" \
  --restore_exclude=global_step,ignore,learning_rate \
  --global_epoch=0 \
  --global_step=0 \
  --learning_rate=0.01 \
  --min_learning_rate=1e-06 \
  --optimizer='bert-adam' \
  --batch_size=512 \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --write_summary \
  --write_metric_summary \
  --freeze_graph_final=0 \
  --vie=0.25 \
  $*

