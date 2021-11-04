folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --model=Model \
  --bert_dir=../input/data/uncased_L-2_H-128_A-2 \
  --doc_dir=../input/data \
  --batch_size=8 \
  --mname=$x \
  $*
