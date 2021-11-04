folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --model=Model \
  --feats=user,doc,day,device,author,feed,song,singer \
  --feats2=manual_keys,machine_keys,manual_tags,machine_tags,desc \
  --his_actions=read_comments,comments,likes,click_avatars,forwards,follows,favorites,negs,poss,finishs,unfinishs \
  --his_id_feats=feed \
  --word_emb=word_emb \
  --doc_emb=doc_emb \
  --user_emb=user_emb \
  --author_emb=author_emb \
  --singer_emb=singer_emb \
  --song_emb=song_emb \
  --tag_emb=tag_emb \
  --mname=$x \
  $*

