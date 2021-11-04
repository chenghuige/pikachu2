folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --model=Model \
  --bert_dir=google/electra-small-discriminator \
  --doc_dir=../input/data \
  --batch_size=8 \
  --lr=5e-5 \
  --min_lr=1e-8 \
  --bert_only=1 \
  --fp16 \
  --mname=$x \
  $*
