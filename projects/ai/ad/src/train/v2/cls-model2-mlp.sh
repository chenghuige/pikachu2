folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --num_layers=1 \
  --model='ClsModel2Mlp' \
  --mname=$x \
  $*
