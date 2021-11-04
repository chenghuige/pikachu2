
echo `date +"%Y%m%d %H:%M:%S"`$'\t'"start...."

tm=$1
interval=$2

source ./config.py

# force to use python 2.7 for spark
export PATH='/usr/bin:'$PATH

product="sgsapp"
types=('tuwen' 'video' 'all')

version=1

# sgsapp and hour is must
model_name=${types[$((MARK))]}"_hour_${product}_v${version}"

# change to your name
user=chenghuige

valid_span=1 # for validation by defualt might be 1 hour ago, here use 2 which means 2 hours to deploy model

dedup=0
filter=0

#------below might change if you want to more debug or more speed
# lr score already in real show feature new but seems only small ratio, if FIXED then no need to join here
join=0
eval=1
del_inter_result=1
# del_inter_result=0

# change it to your own path!
base_feature_dir="/user/traffic_dm/chg/rank/${model_name}"

start=`date -d"${tm:0:8} ${tm:8:10}" +"%Y%m%d %H"`
start_hour=`date -d "$start" +"%Y%m%d%H"`
# one week data saved
del_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -240hours" +"%Y%m%d%H"`
local_del_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -5hours" +"%Y%m%d%H"`


echo "the first start_hour:" ${start_hour}
echo "the second interval:" ${interval}
echo "model_name:" ${model_name}

ugi="traffic_dm,6703eb6d6dad1360d73b045f6d6b0868"
name_password="-D hadoop.client.ugi=${ugi}"
sample_input_dir="hdfs://GodSonNameNode2/user/traffic_dm/fujinbing/real_show_feature_new"
input_show_dir='hdfs://GodSonNameNode2/user/traffic_dm/fujinbing/data'
local_root="/home/gezi/tmp/rank/"
# local_root="."

# if set 0 better random as use like all 24 hours data to random shuffle and generate tfrecords be slower
tfrecords_gen_parallel=0
# queuename=traffic_dm 
queuename=feedflow_online
# for trainning using valid_parts tfrecordss do validation
# valid_parts=10
all_parts=50

# currently not used
interval_json_data="${local_root}/data/interval_data/interval.json"

logfile="${local_root}/${model_name}_${start_hour}_${interval}.log"
del_logfile="${local_root}/${model_name}_${start_hour}_${interval}.log"


#---------------------- for 1 hour gen feature

# notice you need to manuly delete exist dir if you want to rerun complete
gen_feature_exist_flag="${base_feature_dir}/exist/gen_feature/${start_hour}"
deal_features_exist_flag="${base_feature_dir}/exist/deal_features/${start_hour}"

gen_feature_output_dir="${base_feature_dir}/gen_feature"
gen_feature_output="${gen_feature_output_dir}/${start_hour}"
gen_feature_output_utf8="${gen_feature_output_dir}/.utf8/${start_hour}"
gen_feature_output_dedup="${gen_feature_output_dir}/.dedup/${start_hour}"
gen_feature_output_filter="${gen_feature_output_dir}/filter/${start_hour}"
gen_feature_output_join="${gen_feature_output_dir}/.join/${start_hour}"
if (($del_inter_result==1))
then
del_gen_feature_output="${gen_feature_output_dir}/${del_hour}"
else
del_gen_feature_output="${gen_feature_output_dir}/${del_hour} ${gen_feature_output_dir}/*/${del_hour} ${base_feature_dir}/exist/*/${del_hour}"
fi
inter_gen_feature_dirs="${gen_feature_output_dir}/.*/${start_hour}"

eval_output_dir="${base_feature_dir}/eval"
eval_output="${eval_output_dir}/${start_hour}"
del_eval_output="${eval_output_dir}/${del_hour}"

stats_feature_output_dir="${base_feature_dir}/stats_feature"
stats_feature_output="${stats_feature_output_dir}/${start_hour}"
del_stats_feature_output="${stats_feature_output_dir}/${del_hour}"

#------------------------- for nhours merge 
dedup_features_output_dir="${base_feature_dir}/dedup_features"
dedup_features_output="${dedup_features_output_dir}/${start_hour}_${interval}"
del_dedup_features_output="${dedup_features_output_dir}/${del_hour}_*"

filter_features_output_dir="${base_feature_dir}/filter_features"
filter_features_output="${filter_features_output_dir}/${start_hour}_${interval}"
del_filter_features_output="${filter_features_output_dir}/${del_hour}_*"

deal_features_input=${gen_feature_output_dir}
is_interval_dirs=1
if [ $dedup == '1' ];then
deal_features_input=${dedup_features_output}
is_interval_dirs=0
fi
if [ $filter == '1' ];then
deal_features_input=${filter_features_output}
is_interval_dirs=0
fi

deal_eval_input=$eval_output

stats_features_output_dir="${base_feature_dir}/stats_features"
stats_features_output="${stats_features_output_dir}/${start_hour}_${interval}"
del_stats_features_output="${stats_features_output_dir}/${del_hour}_*"

# for TFRecord  add by libowei
tfrecords_output_dir="${base_feature_dir}/tfrecords"
del_tfrecords_output="${base_feature_dir}/tfrecords/${del_hour}* ${base_feature_dir}/tfrecords/.tmp/${del_hour}*"
py3_env_file="/user/traffic_dm/libowei/python3.tgz"
# encoded_feature_index_list="${stats_features_output}/feature/part-00000"
encoded_feature_index_list="${stats_features_output_dir}/${start_hour}_24/feature/part-00000"
encoded_feature_index_list2="${stats_features_output_dir}/${start_hour}_24/feature_ori/"
tfrecords_job_temp_dir="${base_feature_dir}/tfrecords/.tmp"
local_tfrecords_data_dir="${local_root}/data/${model_name}"
tfrecords_debug=0

#----------------------------- store 1h mid-docid and read mid-docid history
gen_feature_output_mdocid="${gen_feature_output_dir}/mdocid/${start_hour}"
#gen_feature_output_mdocid_dir="${gen_feature_output_dir}/mdocid"
gen_feature_output_mdocid_dir="/user/traffic_dm/baili/new_rank/mdocid_history" #constant dir, not need to change
read_mid_history_interval='24'
