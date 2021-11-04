folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=unet \
  --backbone=official_Xception \
  --mname=$x \
  $*
