folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/common.sh \
  --transformer=bert-base-chinese \
  --mname=$x \
  $*
