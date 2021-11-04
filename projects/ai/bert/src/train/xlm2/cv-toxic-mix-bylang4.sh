folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

RECORD_DIR=../input/tfrecords/xlm
sh ./train/$v/common.sh \
    --model=xlm_model \
    --use_lang=1 \
    --ckpt_dir=../working/exps/toxic-mix-lang \
    --train_input=${RECORD_DIR}/validation-bylang \
    --pretrained=../input/tf-xlm-roberta-base \
    --valid_input=${RECORD_DIR}/validation-bylang \
    --do_test=0 \
    --batch_size=32 \
    --max_len=192 \
    --num_gpus=1 \
    --folds=3 \
    --vie=0.5 \
    --write_valid=1 \
    --cv_save_weights=1 \
    --model_dir=../working/exps/$v/$x \
    $*
