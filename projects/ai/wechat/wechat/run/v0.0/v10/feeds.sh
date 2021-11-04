folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/$v/common.sh \
  --model=Model \
  --emb_dim=128 \
  --feats=user,feed \
  --his_feats2=read_comments,comments,likes,click_avatars,forwards,follows,favorites \
  --his_other_feats=feed \
  --his_ids=0 \
  --use_dense \
  --feed_trainable=0 \
  --task_mlp \
  --weight_loss \
  --mname=$x \
  $*
