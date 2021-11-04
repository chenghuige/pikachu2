folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=Model \
  --incl_feats=uid,did,his_id,^cur_title$,^his_title$ \
  --feat_pooling=concat,dot \
  --use_history \
  --use_uid=1 \
  --use_news_info \
  --use_history_info \
  --his_pooling=att \
  --his_simple_pooling=att \
  --use_entity_pretrain \
  --train_entity_emb \
  --use_title \
  --use_word_pretrain \
  --emb_size=100 \
  --batch_size=1024 \
  --mname=$x \
  $*
