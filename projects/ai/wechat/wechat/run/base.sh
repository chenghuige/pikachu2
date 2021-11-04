folder=$(dirname "$0")
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/common.sh \
  --model=Model \
  --mname=$x \
  $*
