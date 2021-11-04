folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model='ClsBaseline5' \
  --batch_size=512 \
  --mname=$x \
  $*
