folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/$v/common.sh \
  --model=Model \
  --emb_dim=128 \
  --feats=user,doc,day,device,author,feed,singer,song \
  --feats2=manual_keys,machine_keys,manual_tags,machine_tags,desc,desc_char \
  --use_dense \
  --feed_trainable \
  --task_mlp \
  --weight_loss \
  --mname=$x \
  $*
