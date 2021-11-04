folder=$(dirname "$0")
v=${folder##*/}

x=$(basename "$0")
echo $x

bin=./train.py
echo $bin

export SM_FRAMEWORK=tf.keras
export PYTHONPATH=..:$PYTHONPATH

$bin \
  --wandb_key=8924d3fe0dc7e0c22006a68932d672f0366b57fd \
  --wandb_group=${v} \
  --wandb_project=naicseg \
  --model_dir="../working/$v/$x" \
  --train_input='../input/quarter/tfrecords/train/*/*' \
  --restore_exclude=global_step,ignore,learning_rate \
  --global_epoch=0 \
  --global_step=0 \
  --fp16=0 \
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
  --batch_size=16 \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --freeze_graph_final=0 \
  --vie=1 \
  --save_interval_epochs=1 \
  --epochs=1 \
  --allow_cpu=0 \
  --async_valid=0 \
  --print_depth=1 \
  --do_test=0 \
  --custom_evaluate=0 \
  --tb_image=1 \
  --num_train=10000 \
  --fold=1 \
  $*

