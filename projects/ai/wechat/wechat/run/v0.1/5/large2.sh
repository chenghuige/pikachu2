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
  --his_actions=read_comments,comments,likes,click_avatars,forwards,follows,favorites,latests,finishs \
  --his_feats=author,singer,song \
  --his_feats2=manual_tags,manual_keys,desc \
  --max_texts=10 \
  --doc_emb=doc_valid_norm_emb \
  --user_emb=user_valid_norm_emb \
  --author_emb=author_valid_norm_emb \
  --singer_emb=singer_valid_norm_emb \
  --song_emb=song_valid_norm_emb \
  --word_emb=word_norm_emb \
  --share_tag_encoder \
  --use_dense \
  --use_doc_his=1 \
  --use_feed_his=0 \
  --feed_trainable=1 \
  --task_mlp \
  --weight_loss \
  --mname=$x \
  $*

