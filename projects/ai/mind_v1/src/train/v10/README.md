try to use day data day 0
sh ./train/v10/din-title-pretrain.sh --mode=test --loop_train=0 --train_input=../input/tfrecords/train-days/0 --valid_input=../input/tfrecords/train-days/6
sh ./train/v10/din-title-pretrain.sh --mode=valid --write_summary=0 --write_metric_summary=0 --input=../input/tfrecords/train-days/0 --valid_input=../input/tfrecords/train-days/6 --loop_train=0 --valid_mask_dids=0

