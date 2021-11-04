folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/dlrm-att-kw.sh \
  --history_strategy=V2 \
  --use_all_type=1 \
  --use_refresh_emb=1 \
  --click_power=1.38 \
  --dur_power=0.62 \
  --model_name=$x \
  $*
    
