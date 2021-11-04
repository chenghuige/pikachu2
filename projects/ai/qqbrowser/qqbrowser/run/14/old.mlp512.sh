folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

# from_logits scale 1w otherwise 1k
sh ./run/$v/common.sh \
  --model=model \
  --use_bert_lr \
  --remove_pred \
  --nextvlad_dropout=0. \
  --fusion_dropout=0. \
  --top_tags=5 \
  --label_strategy=all_tags \
  --num_negs=1000 \
  --from_logits=1 \
  --from_logits_mask=100 \
  --loss_scale=1000 \
  --epochs=6 \
  --decay_epochs=10 \
  --batch_norm=0 \
  --nextvlad_merge \
  --use_se=0 \
  --mlp_dims=512 \
  --activate_last=1 \
  --mname=$x \
  $*

