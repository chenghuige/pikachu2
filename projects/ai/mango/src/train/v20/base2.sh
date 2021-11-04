folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --learning_rate=0.001 \
  --vid_pretrain=../input/all/glove-filter-128/emb.npy \
  --words_pretrain=../input/all/glove-words-128/emb.npy \
  --min_count=0 \
  --use_w2v \
  --words_w2v \
  --stars_w2v=0 \
  --pooling=dot \
  --model=Model \
  --mname=$x \
  $*
