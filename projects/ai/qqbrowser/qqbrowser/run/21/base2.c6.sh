folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base2.sh \
  --continue_version=6 \
  --mname=$x \
  $*

