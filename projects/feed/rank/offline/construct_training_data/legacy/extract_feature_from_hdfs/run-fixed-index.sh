#!/usr/bin/
#coding=gbk

start_hour=`date -d "-2 hours" "+%Y%m%d%H"`

start_hour2='2019092809'
interval2=24

start_hour='2019092910'
interval=24

#start_hour='2019092810'
#start_hour=$1
#interval=1

model_name="sgsapp_chg_hour"

source ./function.sh $start_hour $interval $model_name

product="sgsapp"

# construct_feature... -------------------------------------------------------------
# combined_feature "${sample_input_dir}" "${no_bad_case_mid_output}" "${construct_feature_script}" "${interval_json_data}" "${model_name}"
# hadoop fs ${name_password} -rm -r ${del_no_bad_case_mid_output}

# dedup ---------------------------
dedup_feature_list_new "${no_bad_case_mid_output_dir}" ${interval} ${dedup_feature_output} ${product}
hadoop fs -${name_password} -rm -r ${del_dedup_feature_output}
base_feature_input=" -input ${dedup_feature_output}/${product}"
#
# sample --------------------------
echo "check the input file: base_feature_input:" ${base_feature_input}
filter_bad_case_mid "${base_feature_input}" "${filter_mid_output}" "${filter_bad_case_mid_mapper}" "${filter_bad_case_mid_reducer}" "${model_name}"
hadoop fs ${name_password} -rm -r ${del_filter_mid_output}

# copy to loacl
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"copy filter_mid_output to local... toCheckTime"
rm -rf "${filter_mid_data_dir}/${filter_mid_data}"
hadoop fs ${name_password} -getmerge ${filter_mid_output} "${filter_mid_data_dir}/${filter_mid_data}"
show_err $? "copy filter_mid_output to local"

#filter
echo "check the input file: base_feature_input:" ${base_feature_input}
filter_train_data "${base_feature_input}" "${filter_bad_case_output}" "${filter_train_data_script}" "${filter_mid_data_dir}" "${filter_mid_data}" "${model_name}"
hadoop fs ${name_password} -rm -r ${del_filter_train_data}
base_feature_input=" -input ${filter_bad_case_output}"


# -------------------------------
# construct_sparse_matrix
feature_index_list="${start_hour2}_${interval2}_feature_index_list"
construct_sparse_matrix "${base_feature_input}" "${sparse_matrix_output}" "${construct_sparse_matrix_mapper}" "${feature_dir}" "${feature_index_list}" "${model_name}"

hadoop fs ${name_password} -rm -r ${sparse_matrix_output_del}
# -copyToLocal
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"copy sparse_matrix_output to local...toCheckTime"

# copyToLocal one part for validation in further step
rm -rf "${sparse_matrix_data_dir}/${start_hour}_${interval}"
mkdir -p "${sparse_matrix_data_dir}/${start_hour}_${interval}"
hadoop fs ${name_password} -copyToLocal ${sparse_matrix_output}/part-00024.lzo ${sparse_matrix_data_dir}/${start_hour}_${interval}
show_err $? "copy sparse_matrix_output"

# construct_tfrecord
cat ${feature_dir}/${feature_index_list} | iconv -f gbk -t utf8 > ${encoded_feature_index_list}
construct_tfrecord "${sparse_matrix_output}" "${tfrecord_output}"  "${tfrecord_job_temp}" "${py3_env_file}" "${construct_tfrecord_script}" "${encoded_feature_index_list}" 0 0 0
hadoop fs ${name_password} -rm -r ${tfrecord_output_del}

# -copyToLocal
if [ ! -d ${tfrecord_data_dir}/${start_hour}_${interval} ];then
  mkdir -p ${tfrecord_data_dir}/${start_hour}_${interval}
fi
hadoop fs ${name_password} -copyToLocal ${tfrecord_output}/* ${tfrecord_data_dir}/${start_hour}_${interval}
show_err $? "copy tfrecord"


tmpwatch -afv 10 ${sparse_matrix_data_dir}
tmpwatch -afv 3 ${filter_mid_data_dir}
tmpwatch -afv 10 ${feature_dir}
tmpwatch -afv 1 ${tfrecord_data_dir}

echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"end...toCheckTime"
