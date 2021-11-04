folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/common.sh \
  --model=Model \
  --emb_dim=128 \
  --dense_feats=video_display,finish_rate_mean,stay_rate_mean,read_comment_rate,like_rate,click_avatar_rate,forward_rate,favorite_rate,comment_rate,follow_rate,actions_rate \
  --count_feats=num_shows,num_actions,video_time2 \
  --feats=user,doc,device,day \
  --max_texts=10 \
  --share_tag_encoder \
  --use_dense \
  --feed_trainable=1 \
  --task_mlp \
  --weight_loss \
  --mname=$x \
  $*

