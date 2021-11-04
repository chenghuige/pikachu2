folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=Model \
  --mdrop \
  --transformer=hfl/chinese-electra-180g-base-discriminator \
  --mname=$x \
  $*
