#!/bin/sh
source /root/.bashrc
# psyduck
sansan1
export LANG=zh_CN.UTF-8

export MARK=video
source ./conf/config.sh $MARK

start_hour=$1

if (($METRIC==1))
then
  train_dir=$DIR/tfrecords/${start_hour}
  valid_dir=$train_dir
  num_valid=0
else
 train_dir=$DIR/tfrecords/${start_hour}
 valid_dir=$train_dir
 num_valid=500000
fi

echo 'train_dir' $train_dir
echo 'valid_dir' $valid_dir

abtestid=1

model_dir=$DIR2/model/${abtestid}
version=video.ab${abtestid}.tw-dlrm-att-avgPol-shareKw

sh ./train/libowei/dlrm-att.sh \
  --click_power=1.3,1.7,0.5 \
  --dur_power=0.7,0.6,1.3 \
  --model_mark=${abtestid} \
  --train_loop=0 \
  --valid_hour=${start_hour} \
  --min_train=10000 \
  --model_dir=$model_dir \
  --model_name='' \
  --train_input=$train_dir \
  --valid_input=$valid_dir \
  --num_valid=$num_valid \
  --use_w2v_kw_emb=0  \
  --merge_kw_emb_pooling='avg' \
  --use_distribution_emb=1 \
  --use_merge_kw_emb=1 --use_doc_kw_merge_emb=1 --use_doc_kw_secondary_merge_emb=1 --use_rel_vd_history_kw_merge_emb=1 --use_tw_history_kw_merge_emb=1 --use_vd_history_kw_merge_emb=1 --use_vd_long_term_kw_merge_emb=1 --use_tw_long_term_kw_merge_emb=1 --use_long_search_kw_merge_emb=1  --use_new_search_kw_merge_emb=1 --use_user_kw_merge_emb=1 \
  --masked_fields='IATKW$,IATKWSE$,os$,mobile_brand,mobile_model,long_term,last,MSUB.*,^CRW.*,^ICBRW.+,^ICFRW.+' \
  --mask_mode=regex-excl \
  $*

#MARK=video V=2 sh ./train/v25/dlrm.sh --click_power=1.3 --dur_power=0.7 --model_mark=8 --model_dir=${model_dir} --model_name="" --data_version=2 --loop_train=0 --num_steps=-3 --train_input=/search/odin/libowei/rank/data_v2/video/tfrecord.*
