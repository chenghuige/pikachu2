folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/mid.freeze.sh --mn=$x
sh ./run/$v/mid.decay05.sh --lr=0.0005 --mn=$x --reset_all

