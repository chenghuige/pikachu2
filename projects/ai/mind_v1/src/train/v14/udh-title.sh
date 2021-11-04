folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --model=Model \
  --his_pooling=att \
  --incl_feats=uid,did,his_id,title \
  --excl_feats=enti \
  --mname=$x \
  $*
