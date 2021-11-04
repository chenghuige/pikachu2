folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --mode=valid \
  --load_graph=0 \
  --tta \
  --backbone_weights='' \
  --model=unet.kaggle_salt \
  --mname=$x \
  $*
