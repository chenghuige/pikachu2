folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run-word/$v/common.sh \
  --bs=256 \
  --custom_model \
  --transformer=../input/w2v/sp/256/word.npy \
  --mname=$x \
  $*
