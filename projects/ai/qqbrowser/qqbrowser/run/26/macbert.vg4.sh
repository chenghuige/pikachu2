folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --transformer=hfl/chinese-macbert-base \
  --vlad_groups=4 \
  --mname=$x \
  $*

