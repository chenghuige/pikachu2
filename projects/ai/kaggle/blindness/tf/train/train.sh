data_dir='../input/tfrecords'
python ./train.py \
    --model_dir=../input/model/tf/baseline \
    --train_input=$data_dir'/train.tfrecords,' \
    --valid_input=$data_dir'/valid.tfrecords,' \
    --test_input=$data_dir'/test.tfrecords,' \
    --variable_strategy=cpu \
    --batch_size=32 \
    --batch_size_per_gpu=1 \
    --save_interval_epochs=5 \
    --metric_eval_interval_steps=0 \
    --valid_interval_epochs=1 \
    --inference_interval_epochs=0 \
    --optimizer=momentum \
    --momentum=0.9 \
    --learning_rate=0.1 \
    --learning_rate_decay_factor=0.99 \
    --num_epochs_per_decay=1. \
    --num_epochs=40 \
    --return_dict=1 \
