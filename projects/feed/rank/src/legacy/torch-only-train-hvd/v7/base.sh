v=7
model=WideDeep
python ./torch-only-train-hvd.py \
    --horovod_scale=0 \
    --keras_linear=1 \
    --sparse_emb=1 \
    --simple_parse=1 \
    --valid_multiplier=10 \
    --deep_final_act=0 \
    --mlp_dims=50 \
    --mlp_drop=0.2 \
    --field_emb=1 \
    --pooling=sum \
    --dense_activation=relu \
    --model=$model \
    --num_epochs=2 \
    --eager=0 \
    --first_interval_epoch=0.1 \
    --valid_interval_epochs=0.5 \
    --train_input=../input/lmdb/train \
    --valid_input=../input/lmdb/valid \
    --model_dir=../input/model/v$v/torch.only.hvd/base \
    --batch_size=512 \
    --max_feat_len=100 \
    --optimizer=bert-Adam \
    --optimizer2=bert-Adam \
    --opt_weight_decay=0. \
    --opt_epsilon=1e-6 \
    --min_learning_rate=1e-6 \
    --warmup_steps=1000 \
    --learning_rate=0.001 \
    --learning_rate2=0.1 \
