folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/common.sh \
  --model=Model \
  --emb_dim=128 \
  --dense_feats=video_display,finish_rate_mean,stay_rate_mean \
  --feats=user,doc,device,day \
  --max_texts=10 \
  --share_tag_encoder \
  --use_dense \
  --feed_trainable=1 \
  --task_mlp \
  --weight_loss \
  --mname=$x \
  $*

