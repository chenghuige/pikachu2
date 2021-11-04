folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/dlrm-att-kw.sh \
  --history_strategy=V2 \
  --use_all_type=1 \
  --user_emb_factor=1 \
  --doc_emb_factor=2 \
  --num_feature_buckets=4000000 \
  --model_name=$x \
  $*
    
