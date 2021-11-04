folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/new.sh \
  --contrasive_rate=1. \
  --normalloss_rate=0. \
  --mname=$x \
  $*

