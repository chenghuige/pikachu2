folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --model=Model \
  --his_id_feats=feed \
  --his_actions=poss \
  --mname=$x \
  $*


