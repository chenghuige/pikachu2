folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --use_vision=0 \
  --use_merge=1 \
  --use_se=0 \
  --mname=$x \
  $*

