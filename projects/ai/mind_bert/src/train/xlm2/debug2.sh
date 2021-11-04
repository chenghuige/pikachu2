folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/$v/common2.sh \
    --folds=5 \
    --model_dir=../working/exps/$v/$x \
    $*
