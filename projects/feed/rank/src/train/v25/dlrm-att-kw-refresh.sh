folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/dlrm-att-kw.sh \
  --history_strategy=V2 \
  --use_refresh_emb=1 \
  --click_power=1.39 \
  --dur_power=0.61 \
  --user_emb_factor=1 \
  --doc_emb_factor=2 \
  --model_name=$x \
  $*
    
