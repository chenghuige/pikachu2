folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --epochs=10 \
  --decay_epochs=1 \
  --ft_epochs=10 \
  --ft_decay_epochs=0 \
  --mname=$x \
  $*

