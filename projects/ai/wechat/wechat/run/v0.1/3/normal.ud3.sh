folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/$v/common.sh \
  --model=Model \
  --emb_dim=128 \
  --feats=user,doc,day,device,author,feed,song,singer \
  --feats2=manual_keys,machine_keys,manual_tags,machine_tags,desc \
  --max_texts=10 \
  --user_emb=user_norm_emb \
  --doc_emb=doc_norm_emb2 \
  --author_emb=author_norm_emb \
  --singer_emb=singer_norm_emb \
  --song_emb=song_norm_emb \
  --word_emb='' \
  --share_tag_encoder \
  --use_dense \
  --feed_trainable=1 \
  --task_mlp \
  --weight_loss \
  --mname=$x \
  $*

