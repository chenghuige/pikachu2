folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.w2v.sh \
  --model=Model \
  --feats=user,doc,day,device,author,feed,song,singer \
  --feats2=manual_keys,machine_keys,manual_tags,machine_tags,desc \
  --his_actions=read_comments,comments,likes,click_avatars,forwards,follows,favorites,negs,poss,finishs,unfinishs \
  --his_user_actions=todays \
  --his_id_feats=feed,user \
  --mname=$x \
  $*

