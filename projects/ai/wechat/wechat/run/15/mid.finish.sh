folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base2.sh \
  --model=Model \
  --doc_emb=doc_finish_emb \
  --author_emb=author_finish_emb \
  --singer_emb=singer_finish_emb \
  --song_emb=song_finish_emb \
  --pretrain_day=14 \
  --feats=user,doc,day,device,author,feed,song,singer \
  --feats2=manual_keys,machine_keys,manual_tags,machine_tags,desc \
  --his_actions=read_comments,comments,likes,click_avatars,forwards,follows,favorites,negs,poss,finishs,unfinishs \
  --his_id_feats=feed \
  --mname=$x \
  $*

