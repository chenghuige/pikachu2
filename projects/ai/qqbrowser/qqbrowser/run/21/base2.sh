folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/common.sh \
  --model=model \
  --use_bert_lr \
  --remove_pred=0 \
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
  --layer_norm=1 \
  --use_vision=1 \
  --use_merge=1 \
  --use_se=0 \
  --transformer=hfl/chinese-bert-wwm-ext \
  --continue_pretrain \
  --ft_lr_mul=1. \
  --ft_epochs=5 \
  --decay_epochs=0 \
  --ft_decay_epochs=0 \
  --mname=$x \
  $*

