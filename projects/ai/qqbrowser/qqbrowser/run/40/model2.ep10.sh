folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/model2.sh \
  --ep=10 \
  --mname=$x \
  $*

