folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/model1.sh \
  --model=Model1_1 \
  --mname=$x \
  $*
