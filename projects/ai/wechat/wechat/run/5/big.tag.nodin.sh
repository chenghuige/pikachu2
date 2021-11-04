folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

# using day 15 only missing user emb
sh ./run/$v/base.sh \
  --model=Model \
  --his_pooling=att \
  --feats=user,doc,day,device,author,feed,song,singer \
  --feats2=manual_keys,machine_keys,manual_tags,machine_tags,desc \
  --his_actions=read_comments,comments,likes,click_avatars,forwards,follows,favorites,negs,poss,finishs,unfinishs \
  --his_actions2=read_comments,comments,likes,click_avatars,forwards,follows,favorites,negs \
  --max_his2=10 \
  --his_id_feats=feed \
  --his_feats=author,singer,song \
  --tag_emb=tag_norm_emb \
  --mname=$x \
  $*


