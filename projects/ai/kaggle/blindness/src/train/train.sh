v=0
data_dir='../input/aptos2019-blindness-detection/'
fold=0
python ./train.py \
    --fold=$fold \
    --model=DenseNet121 \
    --model_dir=../input/model/aptos2019-blindness-detection/v$v/tf/baseline \
    --train_input=$data_dir/tfrecords/train/*, \
    --test_input=$data_dir/tfrecords/test/*, \
    --batch_size=32 \
    --save_interval_epochs=2 \
    --save_interval_steps=200 \
    --valid_interval_epochs=1 \
    --optimizer=bert-adam \
    --learning_rate=0.0001 \
    --num_epochs=50 \
    --buffer_size=100 \
    --cache=1 \
    $*
