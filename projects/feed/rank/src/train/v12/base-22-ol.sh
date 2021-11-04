v=12
source ./config2.sh
# for example training data is 2019101321 2hours valid data is 2019101322 1hour
# sh ./train/v10/base.sh 2019101321_2 2019101322_1
#export DIR=$base_dir
train_dir=$1/tfrecords
valid_dir=$DIR/2019102422/tfrecords

echo $train_dir 
echo $valid_dir 

model=WideDeep
python ./train.py \
    --restore_exclude=global_step,ignore,learning_rate \
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
    --valid_interval_epochs=1 \
    --first_interval_epoch=-1 \
    --train_input=$train_dir, \
    --valid_input=$valid_dir, \
    --model_dir=$DIR/model.22.ol \
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
    --min_click_duration=20 \
    --interests_weight=1 \
    --interests_weight_type=clip \
    --min_interests=35 \
    $*
