folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

RECORD_DIR=../input/tfrecords/xlm-pair
sh ./train/$v/common.sh \
    --pretrained=../working/exps/xlm/toxic-mix-unintended/tf-xlm-roberta-base \
    --use_multi_dropout=1 \
    --dropout=0.5 \
    --batch_parse=0 \
    --sampling_rate=0.01 \
    --num_train=10000 \
    --train_input=${RECORD_DIR}/validation,${RECORD_DIR}/jigsaw-toxic-comment-train-google-tr-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-it-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-es-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-ru-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-pt-cleaned,${RECORD_DIR}/jigsaw-toxic-comment-train-google-fr-cleaned \
    --valid_input=${RECORD_DIR}/validation \
    --do_test=0 \
    --batch_size=32 \
    --num_gpus=1 \
    --folds=5 \
    --write_valid=1 \
    --cv_save_weights=1 \
    --model_dir=../working/exps/$v/$x \
    $*
