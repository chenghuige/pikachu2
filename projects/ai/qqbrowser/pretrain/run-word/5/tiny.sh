folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run-word/$v/common.sh \
  --vocab_size=100000 \
  --custom_model \
  --transformer=ckiplab/albert-tiny-chinese \
  --mname=$x \
  $*
