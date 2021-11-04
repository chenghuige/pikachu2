folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

# using day 15 only missing user emb
sh ./run/$v/common.sh \
  --model=Model \
  --emb_dim=128 \
  --task_mlp_dims=512,256,128 \
  --feats=user,doc,day,device,author,feed,song,singer \
  --feats2=manual_keys,machine_keys,manual_tags,machine_tags,desc \
  --his_actions=read_comments,comments,likes,click_avatars,forwards,follows,favorites,negs,poss,finishs,unfinishs \
  --his_actions2=poss,negs \
  --max_his2=20 \
  --his_feats=author,singer,song \
  --share_tag_encoder \
  --word_emb=word_norm_emb \
  --pretrain_day=14.5 \
  --pretrain_day_online=15 \
  --use_dense \
  --feed_trainable=1 \
  --task_mlp \
  --weight_loss \
  --mmoe \
  --mmoe_mlp \
  --mname=$x \
  $*