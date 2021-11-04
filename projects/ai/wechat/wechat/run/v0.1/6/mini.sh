folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/common.sh \
  --model=Model \
  --emb_dim=128 \
  --feats=user,doc \
  --user_emb=user_valid_norm_emb \
  --doc_emb=doc_valid_norm_emb \
  --max_texts=10 \
  --share_tag_encoder \
  --use_dense \
  --feed_trainable=0 \
  --task_mlp \
  --weight_loss \
  --mname=$x \
  $*

