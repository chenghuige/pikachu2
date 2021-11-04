folder=$(dirname "$0")
v=${folder##*/}

x=$(basename "$0")
echo $x

bin=./train.py
if [ $INFER ];then
  if [[ $INFER == "1" ]]
  then
     bin=./infer.py
  fi
fi

if [ $TORCH ];then
  if [[ $TORCH == "1" ]]
  then
     bin=./torch-train.py
  fi
fi

echo $bin

# --restore_exclude=global_step,ignore,learning_rate,OptimizeLoss,Adam \
#  --test_input="../input/tfrecords/test" \
$bin \
  --encoder=LSTM \
  --pooling=mean,max,sum \
  --max_len=128 \
  --emb_size=64 \
  --hidden_size=128 \
  --dropout=0.2 \
  --rdropout=0. \
  --model="Baseline" \
  --model_dir="../working/$v/$x" \
  --train_input="../input/tfrecords3/train" \
  --valid_input="../input/tfrecords3/valid" \
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
  --vie=0.1 \
  $*

