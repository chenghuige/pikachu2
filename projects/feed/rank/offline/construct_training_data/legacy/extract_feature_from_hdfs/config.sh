#!/usr/bin/
#coding=gbk

echo `date +"%Y%m%d %H:%M:%S"`$'\t'"start...toCheckTime"

tm=$1
interval=$2
model_name=$3

base_feature_dir="/user/traffic_dm/chg/rank/feature_info"

start=`date -d"${tm:0:8} ${tm:8:10}" +"%Y%m%d %H"`
start_hour=`date -d "$start" +"%Y%m%d%H"`

echo "the first start_hour:" ${start_hour}
echo "the second interval:" ${interval}
echo "model_name:" ${model_name}
del_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -170hours" +"%Y%m%d%H"`
name_password="-D hadoop.client.ugi=traffic_dm,6703eb6d6dad1360d73b045f6d6b0868"
sample_input_dir="/user/traffic_dm/fujinbing/combined_feature_output_new/"
base_feature_input=""
dedup_input=""

dedup_feature_output="/user/traffic_dm/app_rank/dedup_feature_${model_name}/${start_hour}_${interval}"
del_dedup_feature_output="/user/traffic_dm/app_rank/dedup_feature_${model_name}/${del_hour}_${interval}"
rand_sample_output="/user/traffic_dm/app_rank/rand_sample_${model_name}"

merge_sample_feature="merge_sample_feature.py"
merge_sample_feature_output="${base_feature_dir}_${model_name}/merge_sample_feature_output/${start_hour}_${interval}"


filter_bad_case_mid_mapper="filter_bad_case_mid_mapper_realtime.py"
filter_bad_case_mid_reducer="filter_bad_case_mid_reducer_realtime.py"
filter_mid_output="${base_feature_dir}_${model_name}/filter_mid_output/${start_hour}_${interval}"
del_filter_mid_output="${base_feature_dir}_${model_name}/filter_mid_output/${del_hour}_${interval}"
no_bad_case_mid_output="${base_feature_dir}_${model_name}/combined_feature_output_no_bad_case_mid/${start_hour}"
no_bad_case_mid_output_dir="${base_feature_dir}_${model_name}/combined_feature_output_no_bad_case_mid/"
del_no_bad_case_mid_output="${base_feature_dir}_${model_name}/combined_feature_output_no_bad_case_mid/${del_hour}"
filter_bad_case_output="${base_feature_dir}_${model_name}/filter_bad_case_output/${start_hour}_${interval}"
del_filter_bad_case_output="${base_feature_dir}_${model_name}/filter_bad_case_output/${del_hour}_${interval}"
filter_mid_data_dir="./data/badmid_data_${model_name}"
if [ ! -d $filter_mid_data_dir ];then
    mkdir -p $filter_mid_data_dir
fi
filter_mid_data="${start_hour}_${interval}_badmid.data"

filter_bad_case_docid_mapper="filter_bad_case_docid_mapper_realtime.py"
filter_bad_case_docid_reducer="filter_bad_case_docid_reducer_realtime.py"
filter_docid_output="${base_feature_dir}_${model_name}/filter_docid_output/${start_hour}_${interval}"
del_filter_docid_output="${base_feature_dir}_${model_name}/filter_docid_output/${del_hour}_${interval}"
filter_train_data_script="filter_train_data.py"
no_bad_case_docid_output="${base_feature_dir}_${model_name}/combined_feature_output_no_bad_case_docid/${start_hour}_${interval}"
del_no_bad_case_docid_output="${base_feature_dir}_${model_name}/combined_feature_output_no_bad_case_docid/${del_hour}_${interval}"
filter_docid_data_dir="./data/baddocid_data_${model_name}"
if [ ! -d $filter_docid_data_dir ];then
    mkdir -p $filter_docid_data_dir
fi
filter_docid_data="${start_hour}_${interval}_baddocid.data"

construct_feature_script="filter_bad_case_mid_realtime.py"
interval_json_data="./data/interval_data/interval.json"

collect_feature_list_mapper="collect_feature_list_mapper.py"
collect_feature_list_reducer="collect_feature_list_reducer.py"
feature_list_output="${base_feature_dir}_${model_name}/feature_list_output/${start_hour}_${interval}"
feature_list_output_del="${base_feature_dir}_${model_name}/feature_list_output/${del_hour}_${interval}"

feature_dir="./data/feature_data_${model_name}"
feature_list_data="${feature_dir}/${start_hour}_${interval}_feature_list"
feature_infogain="${feature_dir}/${start_hour}_${interval}_feature_infogain"
feature_index_list="${start_hour}_${interval}_feature_index_list"
feature_meta_list="${start_hour}_${interval}_feature_meta_list"

construct_sparse_matrix_mapper="construct_sparse_matrix_mapper.py"
sparse_matrix_output="${base_feature_dir}_${model_name}/sparse_matrix_output_${model_name}/${start_hour}_${interval}"
sparse_matrix_output_del="${base_feature_dir}_${model_name}/sparse_matrix_output_${model_name}/${del_hour}_${interval}"
sparse_matrix_data_dir="./data/sparse_matrix_data_${model_name}"

# for TFRecord  add by libowei
tfrecord_output="/user/traffic_dm/libowei/tfrecord_${model_name}/${start_hour}_${interval}"
tfrecord_output_del="/user/traffic_dm/libowei/tfrecord_${model_name}/${del_hour}_${interval}"
construct_tfrecord_script="gen-record-hdfs.py"
py3_env_file="/user/traffic_dm/libowei/python3.tgz"
encoded_feature_index_list="${feature_dir}/${start_hour}_${interval}_feature_list_encoded"
tfrecord_job_temp="/user/traffic_dm/libowei/temp_${model_name}/${start_hour}_${interval}"
tfrecord_data_dir="./data/tfrecord_data_${model_name}"

