folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=Model \
  --feat_pooling=concat,dot \
  --use_history \
  --use_uid=1 \
  --use_news_info \
  --use_history_info \
  --his_pooling=din \
  --his_simple_pooling=mean \
  --batch_size=1024 \
  --max_history=200 \
  --mname=$x \
  $*
