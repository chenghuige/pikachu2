folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --use_vision_encoder \
  --vlad_groups=8 \
  --mname=$x \
  $*

