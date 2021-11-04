#!/usr/bin/
alarm()
{
    msg=`echo -e "${model_name}_${start_hour}_${interval}\n"`
    msg=`echo -e "$msg\n"`
    for args in $@
    do
        msg="$msg $args"
    done
    msg="new_data_source "${msg}
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

start_hour=$1
hour=${start_hour:8:10}
month=${start_hour:0:6}
interval=1
deploy=1

abtestid=19

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
cloud_root="/search/odin/publicData/CloudS/baili/new_rank"
base_dir="${root}/data/${model_name}"
cloud_base_dir="${cloud_root}/data/${model_name}"
log_dir="${root}/log"
cloud_log_dir="${cloud_root}/log"
cloud_model_root="${cloud_base_dir}/models/${abtestid}"
cur_model_dir="${cloud_model_root}/${start_hour}"
cloud_info_dir="/search/odin/publicData/CloudS/rank/infos"
mkdir -p ${cur_model_dir}
record_dir=${cloud_base_dir}/tfrecords/${start_hour}
newmse_dir="${cloud_root}/data_newmse/${model_name}"
record_dir_newmse="${newmse_dir}/tfrecords/${start_hour}"
shida_dir="${cloud_root}/data_shida/${model_name}"
record_dir_shida="${shida_dir}/tfrecords/${start_hour}"

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

span=2
ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} +${span}hours" +"%Y%m%d%H"`
valid_record_dir=${cloud_base_dir}/tfrecords/${ts_hour}

info_root="${model_dir}/infos/"
info_dir="${model_dir}/infos/${ts_hour}"
mkdir -p $info_dir

valid()
{
chgenv
python /home/gezi/mine/pikachu/tools/infer.py ${valid_record_dir} ${model_dir} --version=${mark} --ofile=${info_dir}/valid.csv
CUDA_VISIBLE_DEVICES=-1 python /home/gezi/mine/pikachu/tools/eval-all.py ${info_root} --online=0 --max_deal=1 &
chgenv
}

train()
{
  pushd  .
  cd ../../../src 
  export METRIC=0
  chgenv
  time sh ./online-train/${mark}/${abtestid}.sh ${start_hour} --hack_device=cpu --sync_hdfs=0 --write_online_summary=0 --train_only
  popd
}

# TODO valid and train in co... now train has to wait  may be by set log_dir to different path like ./model/valid/log.html
# time valid 
train
time valid 

echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"finish ${start_hour}'s task!!!!!..."
# rsync -av $log_dir/* $cloud_log_dir &
# rsync -avP $info_dir $cloud_data_dir &
# wait

