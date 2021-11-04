folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --model=Model \
  --use_uid=0 \
  --use_body \
  --bert_dir=/home/gezi/data/lm/bert/uncased_L-2_H-128_A-2 \
  --doc_dir=../input/doc4 \
  --mname=$x \
  $*
