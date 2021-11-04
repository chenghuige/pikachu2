start_hour=$1
interval=1
if (($#>1)) 
then
interval=$2
fi

source ./function.sh $start_hour $interval 


tfrecords_output=${tfrecords_output_dir}/${start_hour}
tfrecords_job_temp=${tfrecords_job_temp_dir}/${start_hour}

# gen_record --------------------------- notice encoded feature index list as {stats_feature_output}/feature.txt which is on hdfs
gen_records()
{
  input_dir=${deal_features_input}
  for ((i=0; i<$interval; ++i))
  do
      ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d%H"`
      sample_input="$input_dir/$ts_hour"
      sample_output="${tfrecords_output_dir}/${ts_hour}"
      sample_temp="${tfrecords_job_temp_dir}/${ts_hour}"
      ## not filter 
      # time gen_tfrecords ${sample_input} ${sample_output}  ${sample_temp} ${py3_env_file} none 0 0 0 &
      ## filter by feature index
      time gen_tfrecords ${sample_input} ${sample_output}  ${sample_temp} ${py3_env_file} ${field_index_file} 0 0 0 &
  done
  wait
  for ((i=0; i<$interval; ++i))
  do
      ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d%H"`
      sample_output="${tfrecords_output_dir}/${ts_hour}"
      hadoop fs ${name_password} -touchz ${sample_output}/_SUCCESS
  done
}

time gen_records &
PID_gen_records=$!
wait $PID_gen_records

hadoop fs ${name_password} -rm -r ${del_tfrecords_output}

echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"end....gen tfrecords"
