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
<<!
if [[ $VERSION < 3 ]]
then
  chgenv
  export PATH=/home/gezi/env/anaconda3/bin:/home/gezi/mine/pikachu/tools:/home/gezi/mine/pikachu/tools/bin:/home/gezi/soft/py3env/bin/:/usr/local/bin:/usr/bin:$PATH
  export PYTHONPATH=/home/gezi/mine/pikachu/utils/:/home/gezi/mine/pikachu:$PYTHONPAH
else
  chgenv
  export PATH=/home/gezi/env/anaconda3/bin:/home/gezi/mine/pikachu2/tools:/home/gezi/mine/pikachu2/tools/bin:/home/gezi/soft/py3env/bin/:/usr/local/bin:/usr/bin:$PATH
  export PYTHONPATH=/home/gezi/mine/pikachu2/utils/:/home/gezi/mine/pikachu2:$PYTHONPAH
fi
export CUDA_HOME=/home/gezi/env/cuda
export LD_LIBRARY_PATH=/home/gezi/env/anaconda3/lib/:/home/gezi/env/cuda/lib64:/home/gezi/env/cudnn/lib64:/usr/local/lib:/usr/lib64:$LD_LIBRARY_PATH
. /home/gezi/env/anaconda3/etc/profile.d/conda.sh
!

# psyduck
sansan1

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

# root=/search/odin/libowei/rank
root=/search/odin/mkyuwen/rank_onexp_0612

# cloud_root="/search/odin/publicData/CloudS/libowei/rank_online"
cloud_root="/search/odin/publicData/CloudS/mkyuwen/rank_online"
cloud_info_dir="${cloud_root}/infos"
mkdir -p ${cloud_info_dir}/${mark}/${abtestid}/infos_day
base_dir="${root}/${mark}"
cloud_base_dir="${cloud_root}/data/${model_name}"
log_dir="${root}/log"
cloud_log_dir="${cloud_root}/log"
cloud_model_root="${cloud_base_dir}/models/${abtestid}"
cur_model_dir="${cloud_model_root}/${start_hour}"
mkdir -p ${cur_model_dir}
record_dir=$ROOT/sgsapp/data/${mark}_hour_sgsapp_v1//tfrecords/${start_hour}
record_dir_newmse=$ROOT/newmse/data/${mark}_hour_newmse_v1/tfrecords/${start_hour}
record_dir_shida=$ROOT/shida/data/${mark}_hour_shida_v1/tfrecords/${start_hour}

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
time_info="${log_dir}/time_info.${abtestid}.txt"
score_info="${log_dir}/score_info.${abtestid}.txt"
info_root="${base_dir}/infos/${abtestid}"
info_dir="${info_root}/${start_hour}"
mkdir -p $info_dir
score_info2="${info_dir}/score_info.txt"
important_info="${info_dir}/important_info.txt"

# sync="rsync --update -avP"
sync="scp -r"

span=2
ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${span}hours" +"%Y%m%d%H"`
eval_day=`date -d"${start_hour:0:8} ${start_hour:8:10} -23hours" +"%Y%m%d"`

old_model_dir="${cloud_model_root}/${ts_hour}"
valid_dir=$old_model_dir
if [ ! -f ${valid_dir}/model.pb ];then 
  valid_dir=${model_dir}
fi
if [ ! -f ${valid_dir}/model.map ];then 
  valid_dir=${model_dir}
fi

valid_day()
{
  if [[ ! -f ${info_root}/infos_day/${eval_day}/metrics.csv ]]
  then
    CUDA_VISIBLE_DEVICES=-1 python ../../../src/tools/eval-days.py ${base_dir}/infos --models=${abtestid} --type=online --day=${eval_day} --tfrecord_base 
    CUDA_VISIBLE_DEVICES=-1 python ../../../src/tools/eval-days.py ${base_dir}/infos --models=${abtestid} --type=online --day=${eval_day} --tfrecord_base --product=shida 
    CUDA_VISIBLE_DEVICES=-1 python ../../../src/tools/eval-days.py ${base_dir}/infos --models=${abtestid} --type=online --day=${eval_day} --tfrecord_base --product=newmse
    CUDA_VISIBLE_DEVICES=-1 python ../../../src/tools/eval-days.py ${base_dir}/infos --models=${abtestid} --type=online --day=${eval_day} --tfrecord_base --group_by_impression=1 
    CUDA_VISIBLE_DEVICES=-1 python ../../../src/tools/eval-days.py ${base_dir}/infos --models=${abtestid} --type=online --day=${eval_day} --tfrecord_base --product=shida --group_by_impression=1
    CUDA_VISIBLE_DEVICES=-1 python ../../../src/tools/eval-days.py ${base_dir}/infos --models=${abtestid} --type=online --day=${eval_day} --tfrecord_base --product=newmse --group_by_impression=1
    $sync ${info_root}/infos_day/${eval_day} ${cloud_info_dir}/${mark}/${abtestid}/infos_day 
  fi
}

valid()
{
  # CUDA_VISIBLE_DEVICES=-1 
  CUDA_VISIBLE_DEVICES=-1 python ../../../src/infer/infer.py ${record_dir},${record_dir_newmse},${record_dir_shida} ${valid_dir} --version=${mark} --ofile ${info_dir}/valid.csv --force

  CUDA_VISIBLE_DEVICES=-1 python ../../../src/tools/eval-all.py ${info_dir}/valid.csv --offline_abids=${abtestid},15,16,8 --max_deal=1 --force &
  CUDA_VISIBLE_DEVICES=-1 python ../../../src/tools/eval-all.py ${info_dir}/valid.csv --offline_abids=${abtestid},15,16,8 --product=newmse --max_deal=1 --force &
  CUDA_VISIBLE_DEVICES=-1 python ../../../src/tools/eval-all.py ${info_dir}/valid.csv --offline_abids=${abtestid},15,16,8 --product=shida --max_deal=1 --force &

  wait

  CUDA_VISIBLE_DEVICES=-1 python ../../../src/tools/eval-all.py ${info_dir}/valid.csv --offline_abids=${abtestid},15,16,8 --max_deal=1 --force --group_by_impression=1 &
  CUDA_VISIBLE_DEVICES=-1 python ../../../src/tools/eval-all.py ${info_dir}/valid.csv --offline_abids=${abtestid},15,16,8 --product=newmse --max_deal=1 --force --group_by_impression=1 &
  CUDA_VISIBLE_DEVICES=-1 python ../../../src/tools/eval-all.py ${info_dir}/valid.csv --offline_abids=${abtestid},15,16,8 --product=shida --max_deal=1 --force --group_by_impression=1 &

  wait

  valid_day &
  # inverse_ratio=`python ../../../src/tools/inverse-ratio.py ${info_dir}/sgsapp_metrics_offline.csv ${abtestid}`
  # info_toxiaop="[inverse_ratio_abid${abtestid}] ${inverse_ratio}"
  # alarm $info_toxiaop
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
  # --hack_device=cpu 
  version=`time sh ./online-train/${mark}/${abtestid}.sh ${start_hour} --sync_hdfs=0 --write_online_summary=0 --write_summary=0 --write_metric_summary=0 --interval_steps=0  --valid_interval_steps=0 --train_only --freeze_graph_final --emb_device=cpu --min_free_gpu_mem=5000 --max_used_gpu_mem=3000 | tail -1`

  # like valid but using infer.py for current model
  CUDA_VISIBLE_DEVICES=-1 python infer/infer.py ${record_dir} ${model_dir} --version=${mark} --ofile=${model_dir}/valid.csv --force
  CUDA_VISIBLE_DEVICES=-1 python tools/eval-all.py ${model_dir}/valid.csv --online=0 --max_deal=1 --eval_click=0 --eval_dur=0 --eval_quality=0 --eval_cold=0 --force
  auc=`python tools/check-infer.py ${model_dir}/sgsapp_metrics_offline.csv`

  if [[ $auc < 0 ]]
  then
    show_err 1  "model auc is None, please fix bug.."
  else
    if [[ $auc < 0.6 ]]
    then
      show_err 1 "model auc score is too low: ${auc}"
    else
      echo "auc is ok: ${auc}"
    fi
  fi

  popd
  rm -rf ${model_dir}/train.lock
}

# TODO valid and train in co... now train has to wait  may be by set log_dir to different path like ./model/valid/log.html
# time valid 
time valid &
PID_valid=$!

if (($deploy==1))
then
  tmpwatch -avf 50 "${base_dir}/infos/${abtestid}"
  tmpwatch -avf 50 "${model_dir}"
  # rm -rf $epoch_dir

  time train

  #---------------------------------------------
  # check pb file
  #---------------------------------------------

  CUDA_VISIBLE_DEVICES=-1 python check_pb.py "${model_pb}"
  show_err $? "check model.pb"

  #----------------------------
  # online model
  # - cp the model,feature_dict,feature_list to target directory.
  #-----------------------------

  #if [[ $mark == "video" ]]
  #then
  #  sh -x ./deploy_model.sh  ${model_pb} ${feature_index_field} ${abtestid} ${start_hour} &
  #fi

  sh -x ./deploy_model.sh  ${model_pb} ${abtestid} ${start_hour} &
  show_err $? "deploy model error"

  now_hour=`date +\%Y\%m\%d\%H_\%M`

  echo -e "${start_hour}_${interval}\t${version}\t${now_hour}" >> $time_info
 
  sudo cp -rf ${model_dir}/log.html ${info_dir}
  sudo cp -rf ${model_dir}/log.txt ${info_dir}
  sudo cp -rf ${model_dir}/flags.txt ${info_dir}
  echo ${version} > ${info_dir}/version.txt
  echo ${now_hour} > ${info_dir}/time.txt
  
  sync_model()
  {
    $sync ${model_dir}/model.map ${cur_model_dir} 
    $sync ${model_dir}/checkpoint ${cur_model_dir} 
    $sync ${model_dir}/model.pb* ${cur_model_dir} 
    $sync ${model_dir}/command.txt ${cur_model_dir} 
    $sync ${model_dir}/flags.txt ${cur_model_dir} 
    python cp_model.py ${model_dir} ${cur_model_dir} &
    PID_cp_model=$!
    wait $PID_cp_model
  }

  sync_model &
  PID_sync_model=$!
  wait $PID_sync_model

  wait $PID_valid
  $sync ${info_dir} ${cloud_info_dir}/${mark}/${abtestid} &
  wait
  
fi 

echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"finish ${start_hour}'s task!!!!!..."
# rsync -av $log_dir/* $cloud_log_dir &
# rsync -avP $info_dir $cloud_data_dir &
# wait

