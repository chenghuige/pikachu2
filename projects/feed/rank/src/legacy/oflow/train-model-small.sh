python trainer.py \
    --node_num=1 \
    --device_num=8 \
    --train_data_dir="/home/gezi/new/temp/feed/rank/zjx_data_2/of_train" \
    --data_part_num=128 \
    --batch_size_per_device=512 \
    --max_steps=11359 \
    --loss_print_steps=100 \
    --pretrain_model_path='' \
    --model_save_path='snapshots' \
    --weight_l2=0 \
    --enable_model_split=True
