#!/usr/bin/
#coding=gbk


product="sgsapp"

# construct_feature... -------------------------------------------------------------
combined_feature "${sample_input_dir}" "${no_bad_case_mid_output}" "${construct_feature_script}" "${interval_json_data}" "${model_name}"
hadoop fs ${name_password} -rm -r ${del_no_bad_case_mid_output}

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

#------------------------------------------------------------------------------  
echo "check the input file: base_feature_input:" ${base_feature_input}
echo "check the input file: feature_list_output:" ${feature_list_output}
collect_feature_list "${base_feature_input}" "${feature_list_output}" "${collect_feature_list_mapper}" "${collect_feature_list_reducer}"
# copy to loacl
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"copy feature_list to local...toCheckTime"
rm -rf ${feature_list_data}
echo "check the input file: feature_list_output:" ${feature_list_output}
hadoop fs ${name_password} -getmerge ${feature_list_output} ${feature_list_data}
show_err $? "copy feature_list to local"
hadoop fs ${name_password} -rm -r ${feature_list_output_del}

#--------------------------------------------------------
#if [ $model_name = "sgsapp_ddu_recall_realtime" ];then
   temp_path="mid_${model_name}"
   line=`grep  TOTAL_SAMPLES ${feature_list_data}`
   mkdir $temp_path
   split -l 1000000 $feature_list_data -d $temp_path/temp
   index=0
   for file in ${temp_path}/*
   do
   	echo $file
  	 out=out_$model_name$index
   	echo $line|awk -F ' ' '{print $1 "\t" $2 "\t" $3}'  >> $file
   	cat $file |python script/select_feature_based_infogain_ele.py > $out &
   	(( index=$index+1 ))
   done
   wait
   cat  out_$model_name* > ${feature_infogain}
   rm -rf $temp_path
   rm -rf out_$model_name*
#else
#    echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"select_feature_based_infogain...toCheckTime"
#    cat ${feature_list_data} | python script/select_feature_based_infogain.py > ${feature_infogain}
#    show_err $? "select_feature_based_infogain"
#fi
#---------------------------------------------------------------------
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"index_feature...toCheckTime"
python ./script/index_feature_new.py ${feature_infogain} ${feature_dir}/${feature_index_list} 
show_err $? "index_feature"
# -------------------------------
# construct_sparse_matrix
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
