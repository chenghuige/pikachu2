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

abtestid=$3

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
base_dir=$root/data/$model_name

base_local_dir="/home/gezi/tmp/rank/data"
log_dir="/home/gezi/tmp/rank/log"
cur_model_dir="${base_dir}/models/${abtestid}/${start_hour}"
mkdir -p ${cur_model_dir}
cur_data_dir=$cur_model_dir
cloud_data_dir="/search/odin/publicData/CloudS/chenghuige/rank/data/${model_name}/${abtestid}"
record_dir=${base_dir}/tfrecords/${start_hour}


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


cur_dir="$base_dir"
valid_dir="$base_dir"
model_dir="${cur_dir}/model/${abtestid}"
mkdir -p ${cur_model_dir}
train_log="${cur_model_dir}/log.html"
model_pb="${cur_model_dir}/model.pb"
epoch_dir="${cur_model_dir}/epoch"
feature_index_field="${cur_dir}/feature_index_field"
# feature_index_field="${cur_data_dir}/feature_index_field"
time_info="./log/time_info.${abtestid}.txt"
score_info="./log/score_info.${abtestid}.txt"
score_info2="${cur_data_dir}/score_info.txt"
important_info="${cur_data_dir}/important_info.txt"

if [ ! -d "${record_dir}/_SUCCESS" ];then
  show_err $? "${record_dir}/_SUCCESS not ready"
fi

span=1
ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${span}hours" +"%Y%m%d%H"`

old_model_dir="${base_dir}/models/${abtestid}/${ts_hour}"
valid_dir=$old_model_dir
if [ ! -f ${valid_dir}/model.pb ];then 
  valid_dir=${model_dir}
fi


valid()
{
version='tuwen' 
CUDA_VISIBLE_DEVICES=-1 python ../../../src/infer/infer.py ${record_dir} ${valid_dir}/model.pb --version=${version} > ${cur_data_dir}/scores.tmp 
cat ${cur_data_dir}/scores.tmp  | sort -k 1,1 -k 8,8 -k 9,9 > ${cur_data_dir}/scores 
rm -rf ${cur_data_dir}/scores.tmp

`cat ${cur_data_dir}/scores | awk -F'\t' '{if ($5==4||$5==5||$5==6) print $1,$3,$6}' | python ../../../src/tools/eval.py --version=${version} > ${cur_data_dir}/metrics.online.${abtestid_base}.txt`
metrics1=`python read_log.py ${cur_data_dir}/metrics.online.${abtestid_base}.txt important`
all_metrics=`python read_log.py ${cur_data_dir}/metrics.online.${abtestid_base}.txt all`
echo -e "valid_online_abid${abtestid_base}\t${start_hour}\t${version}\t$all_metrics" >> $score_info
echo -e "valid_online_abid${abtestid_base}\t${start_hour}\t${version}\t$all_metrics" >> $score_info2

`cat ${cur_data_dir}/scores | awk -v abtestid=${abtestid} -F'\t' '{if ($5==abtestid) print $1,$3,$6}' | python ../../../src/tools/eval.py --version=${version} > ${cur_data_dir}/metrics.online.${abtestid}.txt`
metrics2=`python read_log.py ${cur_data_dir}/metrics.online.${abtestid}.txt important`
all_metrics=`python read_log.py ${cur_data_dir}/metrics.online.${abtestid}.txt all`
echo -e "valid_online_abid${abtestid}\t${start_hour}\t${version}\t$all_metrics" >> $score_info
echo -e "valid_online_abid${abtestid}\t${start_hour}\t${version}\t$all_metrics" >> $score_info2

`cat ${cur_data_dir}/scores | awk -F'\t' '{if ($5==4||$5==5||$5==6) print $1,$3,$4}' | python ../../../src/tools/eval.py --version=${version} > ${cur_data_dir}/metrics.offline.${abtestid_base}.txt`
metrics3=`python read_log.py ${cur_data_dir}/metrics.offline.${abtestid_base}.txt important`
all_metrics=`python read_log.py ${cur_data_dir}/metrics.offline.${abtestid_base}.txt all`
echo -e "valid_offline_abid${abtestid_base}\t${start_hour}\t${version}\t$all_metrics" >> $score_info
echo -e "valid_offline_abid${abtestid_base}\t${start_hour}\t${version}\t$all_metrics" >> $score_info2

`cat ${cur_data_dir}/scores | awk -v abtestid=${abtestid} -F'\t' '{if ($5==abtestid) print $1,$3,$4}' | python ../../../src/tools/eval.py --version=${version} > ${cur_data_dir}/metrics.offline.${abtestid}.txt`
metrics4=`python read_log.py ${cur_data_dir}/metrics.offline.${abtestid}.txt important`
all_metrics=`python read_log.py ${cur_data_dir}/metrics.offline.${abtestid}.txt all`
echo -e "valid_offline_abid${abtestid}\t${start_hour}\t${version}\t$all_metrics" >> $score_info
echo -e "valid_offline_abid${abtestid}\t${start_hour}\t${version}\t$all_metrics" >> $score_info2

inverse_ratio=`cat ${cur_data_dir}/scores | awk -v abtestid=${abtestid} -F'\t' '{if ($5==abtestid) print $4,$6}' | python ../../../src/tools/inverse-ratio.py`
echo -e "inverse_ratio_abid${abtestid}\t${start_hour}\t${version}\t$inverse_ratio" >> $score_info
echo -e "inverse_ratio_abid${abtestid}\t${start_hour}\t${version}\t$inverse_ratio" >> $score_info2

info_toxiaop="[valid_online_abid${abtestid_base}] $metrics1 [valid_online_abid${abtestid}] $metrics2 [valid_offline_abid${abtestid}] $metrics4 [valid_offline_abid${abtestid_base}] $metrics3 [inverse_ratio_abid${abtestid}] $inverse_ratio"
echo -e $info_toxiaop >> $important_info
alarm $info_toxiaop

python ../../../src/tools/write-tb.py ./log/tb $score_info2 &
python ../../../src/tools/write-tb.py ./log/tb.latest $score_info 1000 &
}

train()
{
  touch ${model_dir}/train.lock
  pushd  .
  cd ../../../src 
  export METRIC=0
  version=`time sh ./online-train/tuwen/${abtestid}.sh ${start_hour} | tail -1`
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
}

# TODO valid and train in co... now train has to wait  may be by set log_dir to different path like ./model/valid/log.html
# time valid 
time valid &
PID_valid=$!

if (($deploy==1))
then
  tmpwatch -avf 10 "./data/${model_name}"
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

  sh -x ./deploy_model.sh  ${model_pb} ${feature_index_field} ${abtestid} ${deploy_model_name}
  show_err $? "deploy model error"

  now_hour=`date +\%Y\%m\%d\%H_\%M`

  echo -e "${start_hour}_${interval}\t${version}\t${now_hour}" >> $time_info

  wait $PID_valid
  sudo cp -rf ./data/${model_dir}/log.html ${cur_data_dir}
  sudo cp -rf ./data/${model_dir}/log.txt ${cur_data_dir}
  sudo cp -rf ./data/${model_dir}/flags.txt ${cur_data_dir}
  echo ${version} > ${cur_data_dir}/version.txt
  echo ${now_hour} > ${cur_data_dir}/time.txt
  sudo cp -rf ./data/${model_dir}/model.pb* ${cur_data_dir}
  sudo cp -rf ./data/${model_dir}/model.map ${cur_data_dir}
  sudo cp -rf ./data/${model_dir}/checkpoint ${cur_data_dir}
  python cp_model.py ./data/${model_dir} ${cur_data_dir}
  
fi 

echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"finish ${start_hour}'s task!!!!!..."
rsync -avP $log_dir/* $cloud_log_dir &
rsync -avP $cur_data_dir $cloud_data_dir &

