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

$bin \
  --model="Baseline" \
  --model_dir="../working/$v/$x" \
  --train_input="../input/tfrecords/train" \
  --test_input="../input/tfrecords/test" \
  --fold=0 \
  --learning_rate=0.01 \
  --min_learning_rate=1e-06 \
  --optimizer='bert-adam' \
  --batch_size=512 \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --write_summary \
  --write_metric_summary \
  --freeze_graph_final=0 \
  --vie=0.1 \
  $*

