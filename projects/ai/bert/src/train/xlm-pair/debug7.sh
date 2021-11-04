folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

bin=./train.py

sh ./train/$v/common.sh \
    --pretrained=../working/exps/xlm/toxic-mix-unintended/tf-xlm-roberta-base \
    --train_input=${RECORD_DIR}/validation,${RECORD_DIR}/jigsaw-toxic-comment-train-google-tr-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-it-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-es-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-ru-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-pt-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-fr-cleaned \
    --use_multi_dropout=1 \
    --use_word_ids2=1 \
    --write_valid=1 \
    --vie=0.5 \
    --cv_save_weights=1 \
    --folds=5 \
    --model_dir=../working/exps/$v/$x \
    $*
