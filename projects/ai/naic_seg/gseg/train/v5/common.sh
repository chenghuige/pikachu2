folder=$(dirname "$0")
v=${folder##*/}

x=$(basename "$0")
echo $x

bin=./train.py
echo $bin

export SM_FRAMEWORK=tf.keras
export PYTHONPATH=..:$PYTHONPATH

$bin \
  --model_dir="../working/$v/$x" \
  --train_input='../input/tfrecords/train/*/*' \
  --test_input='../input/tfrecords/test/*/*' \
  --restore_exclude=global_step,ignore,learning_rate \
  --global_epoch=0 \
  --global_step=0 \
  --learning_rate=1e-3 \
  --min_learning_rate=1e-5 \
  --lr_decay_power=0.5 \
  --optimizer='bert-adam' \
  --dropout=0.3 \
  --label_smoothing=0.05 \
  --aug_train_image \
  --augment_level=3 \
  --hflip_rate=0.5 \
  --vflip_rate=0.5 \
  --rotate_rate=0.5 \
  --sharpen_rate=0.1 \
  --blur_rate=0.1 \
  --unet_large_filters \
  --inter_activation=swish \
  --backbone_weights=noisy-student \
  --deeplab_large_atrous \
  --batch_size=8 \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --write_summary \
  --write_metric_summary \
  --freeze_graph_final=0 \
  --vie=1 \
  --epochs=5 \
  --allow_cpu=0 \
  --async_valid=0 \
  --print_depth=1 \
  --do_test=0 \
  --num_prefetch_batches=32 \
  --custom_evaluate=0 \
  --fold=1 \
  $*

