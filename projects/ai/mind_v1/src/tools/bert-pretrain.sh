BERT_BASE_DIR=/home/gezi/data/lm/bert/uncased_L-2_H-128_A-2
python /home/gezi/other/bert/run_pretraining.py \
  --input_file=../input/tfrecords/click_titles/* \
  --output_dir=../input/bert-pretrain \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=512 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=25000 \
  --num_warmup_steps=250 \
  --learning_rate=2e-5
