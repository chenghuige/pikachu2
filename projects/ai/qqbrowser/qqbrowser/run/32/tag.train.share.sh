folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --tag_w2v \
  --tag_norm \
  --tag_trainable \
  --share_nextvlad \
  --seed=1025 \
  --mname=$x \
  $*

