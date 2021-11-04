folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/common.sh \
  --window_size=128 \
  --attr=doc \
  --mname=$x \
  $*
