folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

RECORD_DIR=../input/tfrecords/xlm
sh ./train/$v/common.sh \
    --use_lang=1 \
    --pretrained=../working/exps/$v/toxic-mix/tf-xlm-roberta-base \
    --train_input=${RECORD_DIR}/validation-bylang \
    --valid_input=${RECORD_DIR}/validation-bylang \
    --do_test=0 \
    --batch_size=32 \
    --num_gpus=1 \
    --folds=3 \
    --max_len=192 \
    --write_valid=1 \
    --cv_save_weights=1 \
    --model_dir=../working/exps/$v/$x \
    $*
