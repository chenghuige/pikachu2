folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --use_vid \
  --use_history \
  --vid_pretrain=../input/all/glove-128-add-train/emb.npy \
  --his_pooling='' \
  --model=Model \
  --mname=$x \
  $*
