folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --model=Model \
  --incl_feats=uid,did,his_id,bert_title \
  --bert_dir=../input/data/bert-pretrain/models/25000 \
  --doc_dir=../input/data \
  --use_body \
  --mname=$x \
  $*
