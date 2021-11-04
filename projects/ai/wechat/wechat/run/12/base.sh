folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/common.sh \
  --model=Model \
  --feats=user,doc,day,device,author,feed,song,singer \
  --emb_dim=128 \
  --pooling=concat \
  --pooling2=dot2 \
  --batch_norm \
  --mlp_activation=dice \
  --his_pooling=din_dice \
  --task_mlp_dims=512,256,128 \
  --share_tag_encoder \
  --feed_emb=feed_pca_embeddings \
  --word_emb=word_emb \
  --doc_emb=doc_emb \
  --user_emb=user_emb \
  --author_emb=author_emb \
  --singer_emb=singer_emb \
  --song_emb=song_emb \
  --tag_emb=tag_emb \
  --pretrain_day=14.5 \
  --pretrain_day_online=15 \
  --use_dense \
  --feed_trainable=1 \
  --task_mlp \
  --weight_loss \
  --loss_fn=softmax \
  --mmoe \
  --mmoe_mlp \
  --mname=$x \
  $*

