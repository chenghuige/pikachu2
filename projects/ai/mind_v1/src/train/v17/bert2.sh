folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --model=Model \
  --bert_dir=../input/bert-pretrain/models/25000 \
  --doc_dir=../input/doc \
  --use_body \
  --mname=$x \
  $*
