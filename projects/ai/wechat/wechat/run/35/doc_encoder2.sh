folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --model=Model \
  --his_id_feats=feed,doc \
  --pooling=concat,dot3 \
  --pooling2='' \
  --return_seqs \
  --return_seqs_only \
  --doc_his_encoder \
  --mname=$x \
  $*


