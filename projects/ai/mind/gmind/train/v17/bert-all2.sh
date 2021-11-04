folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --model=Model \
  --bert_pooling_seqs \
  --bert_dir=../input/data/bert-pretrain/models/25000 \
  --doc_dir=../input/data \
  --use_body \
  --mname=$x \
  $*
