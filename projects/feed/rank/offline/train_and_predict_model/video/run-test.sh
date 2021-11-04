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
interval=24
deploy=1
if (($# > 1))
then
  if (($2 == 0))
  then
    deploy=0
  else
    interval=$2
  fi
fi

# deploy=0

current_ip=`/sbin/ip addr show eth0|grep inet|grep eth0|awk '{print $2}'|awk -F "/" '{print $1}'`
echo "current_ip ${current_ip}"

model_name="sgsapp_video_wide_deep_hour_wdfield_interest"
model_name2=video_hour_sgsapp_v1

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
base_dir=$root/data/$model_name2

base_local_dir="/home/gezi/tmp/rank/data"
log_dir="/home/gezi/tmp/rank/log"

if [ ! -d "./data" ];then
  ln -s $base_local_dir
fi

if [ ! -d "./log" ];then
  ln -s $log_dir
fi

abtestid_base="4,5,6"
abtestid=15

index=0
train_num=0
valid_num=0

cur_dir="$base_dir"
valid_dir="$base_dir"
train_log="${cur_dir}/model/log.html"
valid_log_dir="${valid_dir}/model/valid"
valid_log="${valid_log_dir}/log.html"
model_pb="${cur_dir}/model/model.pb"
epoch_dir="${cur_dir}/model/epoch"
feature_index_field="${cur_dir}/feature_index_field"
# feature_index_field="${cur_dir}/${start_hour}/feature_index_field"
time_info='./log/time_info.txt'
score_info='./log/score_info.txt'
score_info2="${cur_dir}/${start_hour}/score_info.txt"
important_info="${cur_dir}/${start_hour}/important_info.txt"

if [ ! -d "${cur_dir}/${start_hour}/tfrecords/num_records.txt" ];then
  show_err $? "${cur_dir}/${start_hour}/tfrecords/num_records.txt nor ready"
fi

span=1
ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${span}hours" +"%Y%m%d%H"`
old_model_dir="${cur_dir}/${ts_hour}"

valid()
{
export METRIC=1
pushd  .
cd ../../../src 
ONLINE_RESULT_ONLY=1 ABTESTIDS=$abtestid_base sh ./online-train-video.sh ${start_hour} --log_dir=${valid_log_dir} --model_dir=${old_model_dir}
popd
metrics1=`python read_log.py ${valid_log} important`
all_metrics=`python read_log.py ${valid_log} all`
echo -e "valid_online_abid${abtestid_base}\t${start_hour}\t${version}\t$all_metrics" >> $score_info
echo -e "valid_online_abid${abtestid_base}\t${start_hour}\t${version}\t$all_metrics" >> $score_info2
# send msg to report metrics
# alarm "[valid_online_abid${abtestid_base}] $metrics"

pushd  .
cd ../../../src 
ONLINE_RESULT_ONLY=1 ABTESTIDS=$abtestid sh ./online-train-video.sh ${start_hour} --log_dir=${valid_log_dir} --model_dir=${old_model_dir}
popd
metrics2=`python read_log.py ${valid_log} important`
all_metrics=`python read_log.py ${valid_log} all`
echo -e "valid_online_abid${abtestid}\t${start_hour}\t${version}\t$all_metrics" >> $score_info
echo -e "valid_online_abid${abtestid}\t${start_hour}\t${version}\t$all_metrics" >> $score_info2
# send msg to report metrics
# alarm "[valid_online_abid${abtestid}] $metrics"

pushd  .
cd ../../../src 
NO_ONLINE_RESULT=1 ABTESTIDS=$abtestid sh ./online-train-video.sh ${start_hour} --log_dir=${valid_log_dir} --model_dir=${old_model_dir}
popd
metrics3=`python read_log.py ${valid_log} important`
all_metrics=`python read_log.py ${valid_log} all`
echo -e "valid_offline_abid${abtestid}\t${start_hour}\t${version}\t$all_metrics" >> $score_info
echo -e "valid_offline_abid${abtestid}\t${start_hour}\t${version}\t$all_metrics" >> $score_info2
# send msg to report metrics
# alarm "[valid_offline_abid${abtestid}] $metrics"


pushd  .
cd ../../../src 
NO_ONLINE_RESULT=1 ABTESTIDS=$abtestid_base sh ./online-train-video.sh ${start_hour} --log_dir=${valid_log_dir} --model_dir=${old_model_dir}
popd
metrics4=`python read_log.py ${valid_log} important`
all_metrics=`python read_log.py ${valid_log} all`
echo -e "valid_offline_abid${abtestid_base}\t${start_hour}\t${version}\t$all_metrics" >> $score_info
echo -e "valid_offline_abid${abtestid_base}\t${start_hour}\t${version}\t$all_metrics" >> $score_info2
# send msg to report metrics
# alarm "[valid_offline_abid${abtestid_base}] $metrics"
info_toxiaop="[valid_online_abid${abtestid_base}] $metrics1 [valid_online_abid${abtestid}] $metrics2 [valid_offline_abid${abtestid}] $metrics3 [valid_offline_abid${abtestid_base}] $metrics4"
echo -e $info_toxiaop >> $important_info
alarm $info_toxiaop
export METRIC=0
}

train()
{
  pushd  .
  cd ../../../src 
  export METRIC=0
  version=`time sh ./online-train-video.sh ${start_hour} | tail -1`
  popd

  auc_score=`python read_log.py ${train_log} important`

  if [ ${auc_score} == "None" ];then
    show_err 1 "model auc score is None"
  fi

  min_auc_score=0.55
  if [ $(echo "$auc_score < $min_auc_score"|bc) = 1 ]; then 
    show_err 1 "model auc is too low ${auc_score}"
  fi

  metrics=`python read_log.py ${train_log} important`
  # send msg to report metrics
  alarm "[train] $metrics"
}

# TODO valid and train in co... now train has to wait  may be by set log_dir to different path like ./model/valid/log.html
# time valid 
#time valid &
#PID_valid=$!

if (($deploy==1))
then
  tmpwatch -avf 50 "./data/${model_name2}"
  tmpwatch -avf 24 "./data/${model_name2}/model"
  # rm -rf $epoch_dir

  #time train

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

  #sh -x ./deploy_model.sh  ${model_pb} ${feature_index_field} ${model_name}
  #show_err $? "deploy model error"

  now_hour=`date +\%Y\%m\%d\%H_\%M`

  echo -e "${start_hour}_${interval}\t${version}\t${now_hour}" >> $time_info
  
  #wait $PID_valid
  sudo cp -rf ./data/${model_name2}/model/log.html $base_dir/${start_hour}
  sudo cp -rf ./data/${model_name2}/model/log.txt $base_dir/${start_hour}
  sudo cp -rf ./data/${model_name2}/model/flags.txt $base_dir/${start_hour}
  echo ${version} > $base_dir/${start_hour}/version.txt
  echo ${now_hour} > $base_dir/${start_hour}/time.txt
  sudo cp -rf ./data/${model_name2}/model/model.pb* $base_dir/${start_hour}
  sudo cp -rf ./data/${model_name2}/model/model.map $base_dir/${start_hour}
  sudo cp -rf ./data/${model_name2}/model/checkpoint $base_dir/${start_hour}
  python cp_model.py ./data/${model_name2}/model $base_dir/${start_hour}

fi 
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"finish ${start_hour}'s task!!!!!..."


