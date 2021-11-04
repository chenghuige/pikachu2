folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

bin=./train.py

sh ./train/$v/common.sh \
    --pretrained=../working/exps/xlm/toxic-mix-unintended/tf-xlm-roberta-base \
    --use_word_ids2=1 \
    --use_dot=1 \
    --cv_save_weights=1 \
    --folds=5 \
    --model_dir=../working/exps/$v/$x \
    $*
