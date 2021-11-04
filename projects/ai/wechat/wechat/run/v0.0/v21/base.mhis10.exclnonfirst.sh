folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/$v/common.sh \
  --model=Model \
  --emb_dim=128 \
  --records_name=tfrecords2 \
  --feats=user,doc,day,device,author,feed,song,singer \
  --feats2=manual_keys,machine_keys,manual_tags,machine_tags,desc,desc_char \
  --his_actions=read_comments,comments,likes,click_avatars,forwards,follows,favorites,negs,actions,finishs,unfinishs \
  --his_feats=author,singer,song \
  --his_feats2=manual_tags,manual_keys,desc \
  --max_his2=10 \
  --max_texts=10 \
  --word_emb=word_norm_emb \
  --share_tag_encoder \
  --use_dense \
  --feed_trainable=0 \
  --task_mlp \
  --weight_loss \
  --mname=$x \
  $*

