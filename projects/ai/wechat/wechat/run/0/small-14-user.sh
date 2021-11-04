folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

# using day 15 only missing user emb
sh ./run/$v/common.sh \
  --model=Model \
  --emb_dim=128 \
  --task_mlp_dims=512,128 \
  --feats=user,doc,day,device,author,feed,song,singer \
  --feats2=manual_keys,machine_keys,manual_tags,machine_tags,desc \
  --share_tag_encoder \
  --word_emb=word_norm_emb \
  --user_emb=user_norm_emb \
  --pretrain_day=14 \
  --pretrain_day_online=15 \
  --use_dense \
  --feed_trainable=1 \
  --task_mlp \
  --weight_loss \
  --mname=$x \
  $*

