folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

# from_logits scale 1w otherwise 1k
sh ./run/$v/common.sh \
  --use_bert_lr \
  --remove_pred \
  --top_tags=5 \
  --label_strategy=all_tags \
  --num_negs=1000 \
  --from_logits=0 \
  --loss_scale=1000 \
  --mname=$x \
  $*


