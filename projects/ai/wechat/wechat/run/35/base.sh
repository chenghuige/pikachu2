folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/common.sh \
  --model=Model \
  --feats=user,doc,feed,device,day,author,singer,song,video_time,video_time2 \
  --feats2=manual_tags,machine_tags,manual_keys,machine_keys,desc,ocr,asr \
  --his_actions=poss,read_comments,comments,likes,click_avatars,forwards,follows,favorites,negs,finishs,unfinishs,todays \
  --emb_dim=128 \
  --pooling=concat \
  --pooling2=dot3 \
  --batch_norm \
  --mlp_activation=dice \
  --his_pooling=din_norm_dice \
  --task_mlp_dims=512,256,128 \
  --share_tag_encoder \
  --mean_unk \
  --feed_emb=feed_pca_embeddings \
  --doc_emb=doc_w2v_window128_emb \
  --user_emb=user_w2v_window128_emb \
  --pretrain_day=13.5 \
  --pretrain_day_online=15 \
  --use_dense=1 \
  --feed_trainable=1 \
  --task_mlp \
  --weight_loss \
  --mmoe \
  --mmoe_mlp \
  --mname=$x \
  $*


