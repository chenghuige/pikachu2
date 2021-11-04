folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --mode=valid \
  --load_graph=0 \
  --pretrained_dir=unet.kaggle_salt.50epoch.hvflipaug \
  --tta \
  --model=unet.kaggle_salt \
  --backbone_weights='' \
  --tta_weights=0.45,0.35,0.2 \
  --tta_fns=flip_left_right,flip_up_down \
  --mn=tta-best \
  $*
