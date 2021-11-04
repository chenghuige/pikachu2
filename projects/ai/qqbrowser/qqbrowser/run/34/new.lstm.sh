folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/new.sh \
  --use_vision_encoder \
  --vision_encoder=LSTM \
  --rnn_method=bi \
  --excl vision \
  --merge_method=5 \
  --mname=$x \
  $*

