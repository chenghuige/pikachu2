folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/model.sh \
  --ft_lr_mul=1 \
  --hug=roberta \
  --wes=400 \
  --mname=$x \
  $*

