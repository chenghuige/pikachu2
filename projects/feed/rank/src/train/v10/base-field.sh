v=10
source ./config.sh
# for example training data is 2019101321 2hours valid data is 2019101322 1hour
# sh ./train/v10/base.sh 2019101321_2 2019101322_1
export DIR=$base_dir/$1
train_dir=$DIR/tfrecords
valid_dir=$DIR/../$2_$1/tfrecords

model=WideDeep
python ./train.py \
    --duration_weight=1 \
    --sparse_to_dense=1 \
    --dynamic_pad=1 \
    --simple_parse=0 \
    --valid_multiplier=10 \
    --deep_final_act=0 \
    --mlp_dims=50 \
    --mlp_drop=0. \
    --field_emb=1 \
    --pooling=sum \
    --dense_activation=relu \
    --model=$model \
    --num_epochs=1 \
    --valid_interval_epochs=0.5 \
    --first_interval_epoch=0.1 \
    --train_input=$train_dir/*, \
    --valid_input=$valid_dir/*, \
    --model_dir=$DIR/model/v$v/base.field \
    --batch_size=512 \
    --max_feat_len=100 \
    --optimizers=bert-lazyadam,bert-lazyadam \
    --learning_rates=0.001,0.01 \
    --opt_weight_decay=0. \
    --opt_epsilon=1e-6 \
    --min_learning_rate=1e-6 \
    --warmup_proportion=0.1 \
    --learning_rate=0.001 \
    --write_valid=1 \
    --disable_model_suffix=1 \
    --eval_group=1 \
    $*
