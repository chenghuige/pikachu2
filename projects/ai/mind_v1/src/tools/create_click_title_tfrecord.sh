BERT_BASE_DIR=/home/gezi/data/lm/bert/uncased_L-2_H-128_A-2
python /home/gezi/other/bert/create_pretraining_data.py \
  --input_file=../input/click_titles/click_title_corpus.txt_$1 \
  --output_file=../input/tfrecords/click_titles/tfrecord_$1 \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=2
