folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/common.sh \
  --model=Model \
  --emb_dim=128 \
  --feats=user,doc,feed,author,feed,song,singer,day,device \
  --feats2=manual_keys,machine_keys,manual_tags,machine_tags,desc \
  --his_actions=read_comments,comments,likes,click_avatars,forwards,follows,favorites,latests,finishs \
  --max_texts=10 \
  --doc_emb=doc_valid_norm_emb \
  --user_emb=user_valid_norm_emb \
  --author_emb=author_valid_norm_emb \
  --singer_emb=singer_valid_norm_emb \
  --song_emb=song_valid_norm_emb \
  --word_emb=word_norm_emb \
  --share_tag_encoder \
  --use_dense \
  --feed_trainable=0 \
  --task_mlp \
  --weight_loss \
  --mname=$x \
  $*

