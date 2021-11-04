folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base2.sh \
  --transformer=hfl/chinese-macbert-base \
  --mname=$x \
  $*

