folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/common.sh \
  --transformer=hfl/chinese-roberta-wwm-ext \
  --mname=$x \
  $*
