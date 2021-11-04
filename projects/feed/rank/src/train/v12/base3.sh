v=12
source ./config2.sh
# for example training data is 2019101321 2hours valid data is 2019101322 1hour
# sh ./train/v10/base.sh 2019101321_2 2019101322_1
#export DIR=$base_dir
export DIR=/home/gezi/tmp/rank/data/v2_tuwen_hour_sgsapp_v1
#train_dir=$DIR/2019102415/tfrecords
train_dir=$DIR/2019102415/tfrecords,\
$DIR/2019102414/tfrecords,\
$DIR/2019102413/tfrecords,\
$DIR/2019102412/tfrecords,\
$DIR/2019102411/tfrecords,\
$DIR/2019102410/tfrecords,\
$DIR/2019102409/tfrecords,\
$DIR/2019102408/tfrecords,\
$DIR/2019102407/tfrecords,\
$DIR/2019102406/tfrecords,\
$DIR/2019102405/tfrecords,\
$DIR/2019102404/tfrecords,\
$DIR/2019102403/tfrecords,\
$DIR/2019102402/tfrecords,\
$DIR/2019102401/tfrecords,\
$DIR/2019102400/tfrecords,\
$DIR/2019102323/tfrecords,\
$DIR/2019102322/tfrecords,\
$DIR/2019102321/tfrecords,\
$DIR/2019102320/tfrecords,\
$DIR/2019102319/tfrecords,\
$DIR/2019102318/tfrecords,\
$DIR/2019102317/tfrecords,\
$DIR/2019102316/tfrecords
valid_dir=$DIR/2019102416/tfrecords

echo $train_dir 
echo $valid_dir 

model=WideDeep
python ./train.py \
    --buffer_size=1000000 \
    --hash_encoding=1 \
    --feature_dict_size=30000000 \
    --num_feature_buckets=3000000 \
    --field_dict_size=10000 \
    --duration_weight=1 \
    --sparse_to_dense=1 \
    --dynamic_pad=1 \
    --simple_parse=0 \
    --valid_multiplier=1 \
    --deep_final_act=0 \
    --mlp_dims=50 \
    --mlp_drop=0. \
    --field_emb=0 \
    --pooling=sum \
    --dense_activation=relu \
    --model=$model \
    --num_epochs=1 \
    --valid_interval_epochs=0.1 \
    --first_interval_epoch=-1 \
    --train_input=$train_dir, \
    --valid_input=$valid_dir, \
    --model_dir=$DIR/model3 \
    --batch_size=512 \
    --max_feat_len=100 \
    --optimizers=bert-lazyadam,bert-lazyadam \
    --learning_rates=0.001,0.01 \
    --opt_weight_decay=0. \
    --opt_epsilon=1e-6 \
    --min_learning_rate=1e-6 \
    --warmup_proportion=0.1 \
    --learning_rate=0.001 \
    --write_valid=0 \
    --disable_model_suffix=1 \
    --eval_group=1 \
    --min_click_duration=30 \
    $*
