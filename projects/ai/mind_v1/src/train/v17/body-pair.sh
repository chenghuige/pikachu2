folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/body.sh \
  --model=Model \
  --loss_type=pair \
  --mname=$x \
  $*
