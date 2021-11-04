folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --model=Model \
  --mean_unk \
  --feats=user,doc \
  --emb_dim=128 \
  --pooling=concat \
  --pooling2=dot3 \
  --batch_norm \
  --mlp_activation=dice \
  --his_pooling=din_dice \
  --task_mlp_dims=512,256,128 \
  --share_tag_encoder \
  --feed_emb=feed_pca_embeddings \
  --doc_emb=doc_w2v_window8_emb \
  --pretrain_day=13 \
  --pretrain_day_online=15 \
  --use_dense=0 \
  --feed_trainable=1 \
  --task_mlp \
  --weight_loss \
  --mmoe \
  --mmoe_mlp \
  --mname=$x \
  $*


