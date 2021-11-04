folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/c6.sh \
  --activation=dice \
  --mname=$x \
  $*

