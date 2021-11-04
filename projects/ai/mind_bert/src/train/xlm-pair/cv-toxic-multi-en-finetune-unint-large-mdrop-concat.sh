folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

RECORD_DIR=../input/tfrecords/xlm-pair-concat
sh ./train/$v/common.sh \
    --pretrained=../working/exps/v2/toxic-multi-en-finetune-unint-large/tf-xlm-roberta-large \
    --train_input=${RECORD_DIR}/validation \
    --valid_input=${RECORD_DIR}/validation \
    --do_test=1 \
    --use_word_ids2=0 \
    --batch_size=8 \
    --num_gpus=4 \
    --folds=5 \
    --write_valid=1 \
    --cv_save_weights=1 \
    --vie=0.5 \
    --use_multi_dropout=1 \
    --dropout=0.5 \
    --max_len=384 \
    --model_dir=../working/exps/$v/$x \
    $*
