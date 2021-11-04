folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

bin=./train.py

sh ./train/$v/common.sh \
    --pretrained=../working/exps/xlm/toxic-from-mix/tf-xlm-roberta-base \
    --use_multi_dropout=1 \
    --use_word_ids2=1 \
    --write_valid=1 \
    --vie=0.5 \
    --cv_save_weights=1 \
    --folds=5 \
    --model_dir=../working/exps/$v/$x \
    $*
