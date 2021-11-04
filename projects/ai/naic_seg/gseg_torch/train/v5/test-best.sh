folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/eval-best.sh \
  --mode=test \
  --mname=$x \
  $*
