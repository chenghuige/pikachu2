#!/usr/bin/
#coding=gbk

start_hour=`date -d "-2 hours" "+%Y%m%d%H"`

#interval=$((24*14))
interval=$((24*2))

model_name="sgsapp_chg_hour"

source ./function.sh $start_hour $interval $model_name

product="sgsapp"

# construct_feature... -------------------------------------------------------------
for (( i=0; i<=${interval}; ++i ))
do
    k=$(($i+2))
    start_hour=`date -d "-$k hours" "+%Y%m%d%H"`
    echo $interval, $start_hour 
    source  ./function.sh $start_hour $interval $model_name
    combined_feature "${sample_input_dir}" "${no_bad_case_mid_output}" "${construct_feature_script}" "${interval_json_data}" "${model_name}" 2>&1 /tmp/construct.$interval.log &
    hadoop fs ${name_password} -rm -r ${del_no_bad_case_mid_output} 2>&1 /tmp/tmp.log &
done
