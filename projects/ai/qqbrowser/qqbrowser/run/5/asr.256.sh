folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --model=model \
  --use_asr \
  --asr_len=256 \
  --mname=$x \
  $*

