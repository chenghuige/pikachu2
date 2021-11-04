folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

CUSTOM=1 sh ./train/$v/common.sh \
    --folds=5 \
    --model_dir=../working/exps/$v/$x \
    $*
