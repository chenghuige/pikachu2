folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --lm_target='ad_ids' \
  --model='ClsModel2' \
  --mname=$x \
  $*
