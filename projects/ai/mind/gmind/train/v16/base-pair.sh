folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --model=Model \
  --loss_type=pair2 \
  --num_pairs=10 \
  --mname=$x \
  $*
