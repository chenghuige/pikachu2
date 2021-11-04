folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --model=Model \
  --bert_dir=/home/gezi/data/lm/bert/uncased_L-4_H-128_A-2 \
  --doc_dir=../input/doc2 \
  --batch_size=512 \
  --mname=$x \
  $*
