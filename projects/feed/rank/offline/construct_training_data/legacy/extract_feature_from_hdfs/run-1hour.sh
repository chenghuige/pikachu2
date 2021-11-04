#!/usr/bin/
#coding=gbk

start_hour=`date -d "-2 hours" "+%Y%m%d%H"`
interval=48
model_name="sgsapp_wdfield_interest_chg_hour"

source ./function.sh $start_hour $interval $model_name

product="sgsapp"

# construct_feature... -------------------------------------------------------------
for (( i=0; i<=${interval}; ++i ))
do
    echo $interval, $start_hour 
    # combined_feature "${sample_input_dir}" "${no_bad_case_mid_output}" "${construct_feature_script}" "${interval_json_data}" "${model_name}"
    # hadoop fs ${name_password} -rm -r ${del_no_bad_case_mid_output}
    # ts_day=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d"`
    # ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d%H"`
    # sample_input="$sample_input_dir/${ts_hour}/"
    # hadoop fs ${name_password} -test -e ${sample_input}
    # if [ $? -eq 0 ];then
    #     dedup_input=" ${sample_input}${dedup_input}"
    # fi
done