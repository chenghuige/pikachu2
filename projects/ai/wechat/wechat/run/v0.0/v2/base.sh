folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/$v/common.sh \
  --model=Model \
  --feats=user,doc,day,device,author,feed,singer,song \
  --use_dense \
  --feed_trainable \
  --mname=$x \
  $*
