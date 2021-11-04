#!/usr/bin/
alarm()
{
    msg=`echo -e "${model_name}_${start_hour}_${interval}\n"`
    msg=`echo -e "$msg\n"`
    for args in $@
    do
        msg="$msg $args"
    done

    sh ../../common/send_xiaop.sh "${msg}"
}
# ok
show_err()
{
if [[ $1 != 0 ]];then
    msg=`echo -e "Error occured!\n"$2`
    echo ${msg}
    alarm ${msg}
    exit -1
fi
}

source ./config.sh
source /root/.bashrc
chgenv
export PATH=/home/gezi/env/anaconda3/bin:/home/gezi/mine/pikachu/tools:/home/gezi/mine/pikachu/tools/bin:/home/gezi/soft/py3env/bin/:/usr/local/bin:/usr/bin:$PATH
export PYTHONPATH=/home/gezi/mine/pikachu/utils/:/home/gezi/mine/pikachu:$PYTHONPAH
export CUDA_HOME=/usr/local/cuda-10.0-cudnn-7.5/
export LD_LIBRARY_PATH=/home/gezi/env/anaconda3/lib/:/usr/local/cuda-10.0-cudnn-7.5/lib64:/usr/local/lib:/usr/lib64:$LD_LIBRARY_PATH
. /home/gezi/env/anaconda3/etc/profile.d/conda.sh

start_hour=$1
hour=${start_hour:8:10}
month=${start_hour:0:6}
interval=1
deploy=1

if (($#<2)) 
then
echo 'You need to input abtestid'
exit 1
fi

abtestid=$2

# deploy=0

current_ip=`/sbin/ip addr show eth0|grep inet|grep eth0|awk '{print $2}'|awk -F "/" '{print $1}'`
echo "current_ip ${current_ip}"

if [ ${model_name} = "" ];
then
    echo "need model name"
    exit
fi

if [ *"test"* = $model_name ];then
    echo "need model name with no word test"
    exit
fi

min_train_num=10000

root=/home/gezi/tmp/rank
cloud_root="/search/odin/publicData/CloudS/chenghuige/rank"
# root="/search/odin/localData/CloudS/chenghuige/rank"
base_dir="${root}/data/${model_name}"
cloud_base_dir="${cloud_root}/data/${model_name}"
log_dir="${root}/log"
cloud_log_dir="${cloud_root}/log"
cloud_model_root="${cloud_base_dir}/models/${abtestid}"
cur_model_dir="${cloud_model_root}/${start_hour}"
mkdir -p ${cur_model_dir}
record_dir=${cloud_base_dir}/tfrecords/${start_hour}

if [ ! -d "./data" ];then
  ln -s $base_local_dir
fi

if [ ! -d "./log" ];then
  ln -s $log_dir
fi

abtestid_base="4,5,6"

index=0
train_num=0
valid_num=0


model_dir="${base_dir}/model/${abtestid}"
mkdir -p ${model_dir}
train_log="${model_dir}/log.html"
model_pb="${model_dir}/model.pb"
epoch_dir="${cur_model_dir}/epoch"
feature_index_field="${base_dir}/feature_indexes/feature_index_field"
time_info="${log_dir}/time_info.${abtestid}.txt"
score_info="${log_dir}/score_info.${abtestid}.txt"
info_dir="${base_dir}/infos/${abtestid}/${start_hour}"
mkdir -p $info_dir
score_info2="${info_dir}/score_info.txt"
important_info="${info_dir}/important_info.txt"

# if [ ! -f "${record_dir}/_SUCCESS" ];then
#   show_err $? "${record_dir}/_SUCCESS not ready"
# fi

span=1
ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${span}hours" +"%Y%m%d%H"`

old_model_dir="${cloud_model_root}/${ts_hour}"
valid_dir=$old_model_dir
if [ ! -f ${valid_dir}/model.pb ];then 
  valid_dir=${model_dir}
fi
if [ ! -f ${valid_dir}/model.map ];then 
  valid_dir=${model_dir}
fi

valid()
{
CUDA_VISIBLE_DEVICES=-1 python ../../../src/infer/infer.py ${record_dir} ${valid_dir} --version=${mark} > ${info_dir}/scores.tmp 
cat ${info_dir}/scores.tmp  | sort -k 1,1 -k 8,8 -k 9,9 > ${info_dir}/scores 
rm -rf ${info_dir}/scores.tmp

`cat ${info_dir}/scores | awk -F'\t' '{if (($5==4||$5==5||$5==6) && !($13==931||$13==984||$13==925||$13==926)) print $1,$3,$6}' | python ../../../src/tools/eval.py --version=${mark} > ${info_dir}/metrics.online.${abtestid_base}.txt`
metrics1=`python read_log.py ${info_dir}/metrics.online.${abtestid_base}.txt important`
all_metrics=`python read_log.py ${info_dir}/metrics.online.${abtestid_base}.txt all`
echo -e "valid_online_abid${abtestid_base}\t${start_hour}\t${mark}\t$all_metrics" >> $score_info
echo -e "valid_online_abid${abtestid_base}\t${start_hour}\t${mark}\t$all_metrics" >> $score_info2

`cat ${info_dir}/scores | awk -v abtestid=${abtestid} -F'\t' '{if (($5==abtestid) && !($13==931||$13==984||$13==925||$13==926)) print $1,$3,$6}' | python ../../../src/tools/eval.py --version=${mark} > ${info_dir}/metrics.online.${abtestid}.txt`
metrics2=`python read_log.py ${info_dir}/metrics.online.${abtestid}.txt important`
all_metrics=`python read_log.py ${info_dir}/metrics.online.${abtestid}.txt all`
echo -e "valid_online_abid${abtestid}\t${start_hour}\t${mark}\t$all_metrics" >> $score_info
echo -e "valid_online_abid${abtestid}\t${start_hour}\t${mark}\t$all_metrics" >> $score_info2

`cat ${info_dir}/scores | awk -F'\t' '{if (($5==4||$5==5||$5==6) && !($13==931||$13==984||$13==925||$13==926)) print $1,$3,$4}' | python ../../../src/tools/eval.py --version=${mark} > ${info_dir}/metrics.offline.${abtestid_base}.txt`
metrics3=`python read_log.py ${info_dir}/metrics.offline.${abtestid_base}.txt important`
all_metrics=`python read_log.py ${info_dir}/metrics.offline.${abtestid_base}.txt all`
echo -e "valid_offline_abid${abtestid_base}\t${start_hour}\t${mark}\t$all_metrics" >> $score_info
echo -e "valid_offline_abid${abtestid_base}\t${start_hour}\t${mark}\t$all_metrics" >> $score_info2

`cat ${info_dir}/scores | awk -v abtestid=${abtestid} -F'\t' '{if (($5==abtestid) && !($13==931||$13==984||$13==925||$13==926)) print $1,$3,$4}' | python ../../../src/tools/eval.py --version=${mark} > ${info_dir}/metrics.offline.${abtestid}.txt`
metrics4=`python read_log.py ${info_dir}/metrics.offline.${abtestid}.txt important`
all_metrics=`python read_log.py ${info_dir}/metrics.offline.${abtestid}.txt all`
echo -e "valid_offline_abid${abtestid}\t${start_hour}\t${mark}\t$all_metrics" >> $score_info
echo -e "valid_offline_abid${abtestid}\t${start_hour}\t${mark}\t$all_metrics" >> $score_info2

inverse_ratio=`cat ${info_dir}/scores | awk -v abtestid=${abtestid} -F'\t' '{if (($5==abtestid) && !($13==931||$13==984||$13==925||$13==926)) print $4,$6}' | python ../../../src/tools/inverse-ratio.py`
echo -e "inverse_ratio_abid${abtestid}\t${start_hour}\t${mark}\t$inverse_ratio" >> $score_info
echo -e "inverse_ratio_abid${abtestid}\t${start_hour}\t${mark}\t$inverse_ratio" >> $score_info2

info_toxiaop="[valid_online_abid${abtestid_base}] $metrics1 [valid_online_abid${abtestid}] $metrics2 [valid_offline_abid${abtestid}] $metrics4 [valid_offline_abid${abtestid_base}] $metrics3 [inverse_ratio_abid${abtestid}] $inverse_ratio"
echo -e $info_toxiaop >> $important_info
alarm $info_toxiaop

python ../../../src/tools/write-tb.py ${log_dir}/tb $score_info2 &
python ../../../src/tools/write-tb.py ${log_dir}/tb.latest $score_info 1000 &
}

train()
{
  if [[ -f $${model_dir}/train.lock ]]; then
    sleep 20m
  fi

  touch ${model_dir}/train.lock
  pushd  .
  cd ../../../src 
  export METRIC=0
  version=`time sh ./online-train/${mark}/${abtestid}.sh ${start_hour} --hack_device=cpu --sync_hdfs=0 --write_online_summary=0 --train_only=1 | tail -1`
  popd

  auc_score=`python read_log.py ${train_log} auc`

  if [ ${auc_score} == "None" ];then
    show_err 1 "model auc score is None"
  fi

  min_auc_score=0.4
  if [ $(echo "$auc_score < $min_auc_score"|bc) = 1 ]; then 
    show_err 1 "model auc is too low ${auc_score}"
  fi

  metrics=`python read_log.py ${train_log} important`
  # send msg to report metrics
  alarm "[train] $metrics"]
  rm -rf ${model_dir}/train.lock
  # not use tensorboard
  # rm -rf ${model_dir}/events* 
}

# TODO valid and train in co... now train has to wait  may be by set log_dir to different path like ./model/valid/log.html
# time valid 
#time valid &
#PID_valid=$!

if (($deploy==1))
then
  tmpwatch -avf 10 "${base_dir}/feature_indexes"
  tmpwatch -avf 240 "${base_dir}/infos/${abtestid}"
  tmpwatch -avf 240 "${model_dir}"
  # rm -rf $epoch_dir

  time train

  #---------------------------------------------
  # check pb file
  #---------------------------------------------

  CUDA_VISIBLE_DEVICES=-1 python check_pb.py "${model_pb}"
  show_err $? "check model.pb"

  num_lines=`wc -l ${feature_index_field} | awk '{print $1}'`
  if (($num_lines < 2000000))
  then 
  echo "bad feature index ${feature_index_field} with num_lines $num_lines"
  exit 1
  fi

  #----------------------------
  # online model
  # - cp the model,feature_dict,feature_list to target directory.
  #-----------------------------

  # if [[ $mark == "tuwen" ]]
  # then
  #   if [[ $abtestid == 16 ]]
  #   then
  #     sh -x ./deploy_model.sh  ${model_pb} ${feature_index_field} ${abtestid} ${deploy_model_name} &
  #     sh -x ./deploy_model.sh  ${model_pb} ${feature_index_field} 15 ${deploy_model_name} &
  #   fi
  # else
  #   if [[ $abtestid == 15 ]]
  #   then
  #     sh -x ./deploy_model.sh  ${model_pb} ${feature_index_field} ${abtestid} ${deploy_model_name} &
  #     sh -x ./deploy_model.sh  ${model_pb} ${feature_index_field} 16 ${deploy_model_name} &
  #   fi
  # fi

  #sh -x ./deploy_model.sh  ${model_pb} ${feature_index_field} ${abtestid} ${deploy_model_name} &
  
  #show_err $? "deploy model error"

  now_hour=`date +\%Y\%m\%d\%H_\%M`

  echo -e "${start_hour}_${interval}\t${version}\t${now_hour}" >> $time_info
 
  sudo cp -rf ${model_dir}/log.html ${info_dir}
  sudo cp -rf ${model_dir}/log.txt ${info_dir}
  sudo cp -rf ${model_dir}/flags.txt ${info_dir}
  echo ${version} > ${info_dir}/version.txt
  echo ${now_hour} > ${info_dir}/time.txt
  
  sync_model()
  {
    sudo rsync -avP ${model_dir}/model.map ${cur_model_dir} 
    sudo rsync -avP ${model_dir}/checkpoint ${cur_model_dir} 
    sudo rsync -avP ${model_dir}/model.pb* ${cur_model_dir} 
    python cp_model.py ${model_dir} ${cur_model_dir} &
    PID_cp_model=$!
    wait $PID_cp_model
  }

  sync_model &
  PID_sync_model=$!
  wait $PID_sync_model
  
fi 

echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"finish ${start_hour}'s task!!!!!..."
# rsync -av $log_dir/* $cloud_log_dir &
# rsync -avP $info_dir $cloud_data_dir &
# wait

