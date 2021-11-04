folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

bin=./train.py

sh ./train/$v/base.sh \
    --multi_head=1 \
    --model_dir=../working/exps/$v/$x \
    $*
