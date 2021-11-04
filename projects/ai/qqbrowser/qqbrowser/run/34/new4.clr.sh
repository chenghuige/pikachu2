folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/new4.sh \
   --contrasive_rate=0.1 \
  --mname=$x \
  $*

