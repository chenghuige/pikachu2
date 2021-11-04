folder=$(dirname "$0")
v=${folder##*/}

x=$(basename "$0")
echo $x

bin=./train.py
echo $bin

#  --num_valid=1000000 \
#  --valid_input='../input/tfrecords/dev/*7.*,../input/tfrecords/dev/*8.*,../input/tfrecords/dev/*9.*' \
$bin \
  --model="Model" \
  --model_dir="../working/$v/$x" \
  --train_input='../input/tfrecords/padded/train-0' \
  --valid_input='../input/tfrecords/padded/dev' \
  --test_input='../input/tfrecords/padded/test' \
  --record_padded=0 \
  --fixed_pad=0 \
  --static_input \
  --functional_model=0 \
  --model='Model' \
  --restore_exclude=global_step,ignore,learning_rate \
  --global_epoch=0 \
  --global_step=0 \
  --learning_rate=0.001 \
  --min_learning_rate=1e-06 \
  --optimizer='bert-adam' \
  --batch_size=128 \
  --interval_steps=100 \
  --valid_interval_steps=100 \
  --save_interval_steps=100000000000 \
  --write_summary \
  --write_metric_summary \
  --write_valid_final \
  --freeze_graph_final=0 \
  --allow_cpu=0 \
  --async_valid \
  --max_history=200 \
  --max_titles=50 \
  --nvs=4 \
  --num_valid=1000000 \
  --shuffle=0 \
  --write_valid \
  --valid_mask_dids \
  --test_mask_dids=0 \
  --train_mask_dids=0 \
  --test_all_mask=0 \
  --mask_dids_ratio=0.92 \
  --input_dir='../input/data' \
  --doc_dir='../input/data' \
  --entity_pretrain='../input/data/entity_emb2.npy' \
  --word_pretrain='../input/data/glove-100/emb.npy' \
  --title_lookup='../input/data/title_lookup.npy' \
  --doc_lookup='../input/data/doc_lookup.npy' \
  --doc_fnames='../input/data/doc_fnames.npy' \
  --doc_flens='../input/data/doc_flens.npy' \
  $*

