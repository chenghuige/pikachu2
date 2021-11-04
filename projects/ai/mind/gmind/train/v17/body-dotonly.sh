folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --model=Model \
  --use_body \
  --doc_dir=../input/doc3 \
  --feat_pooling=dot \
  --mname=$x \
  $*
