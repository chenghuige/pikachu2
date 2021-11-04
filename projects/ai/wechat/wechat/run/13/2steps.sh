folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/$1.sh --bert_style_lr=0 --embs_trainable=0 --mn=$1.$x
sh ./run/$v/$1.sh --mn=$1.$x --reset_all

