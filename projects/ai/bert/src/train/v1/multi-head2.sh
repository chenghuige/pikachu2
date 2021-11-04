folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

bin=./train.py

sh ./train/$v/multi-head.sh \
    --opt_epsilon=1e-6 \
    --model_dir=../working/exps/$v/$x \
    $*
