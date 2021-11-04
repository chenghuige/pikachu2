folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/new.sh \
  --gpus=4 \
  --ft_bs_mul=0.5 \
  --transformer=hfl/chinese-macbert-large \
  --mname=$x \
  $*

