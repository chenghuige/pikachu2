folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.w2v.sh \
  --model=Model \
  --feats=user,doc,day,device,author,feed,song,singer,desc_vec,ocr_vec,asr_vec \
  --feats2=manual_keys,machine_keys,manual_tags,machine_tags,desc \
  --his_actions=read_comments,comments,likes,click_avatars,forwards,follows,favorites,negs,poss,finishs,unfinishs \
  --his_id_feats=feed,desc_vec,ocr_vec,asr_vec \
  --din_keys=feed \
  --desc_vec_emb=desc_vec_emb \
  --ocr_vec_emb=ocr_vec_emb \
  --asr_vec_emb=asr_vec_emb \
  --mname=$x \
  $*

