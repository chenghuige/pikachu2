folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=EfficientDet \
  --backbone=EfficientNetB5 \
  --image_size=512,512 \
  --batch_size=16 \
  --mname=$x \
  $*
