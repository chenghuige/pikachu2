folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/new2.sh \
  --transformer=bert-base-chinese \
  --mname=$x \
  $*

