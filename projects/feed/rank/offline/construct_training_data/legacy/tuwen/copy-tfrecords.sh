start_hour=$1
interval=1
if (($#>1)) 
then
interval=$2
fi


source ./function.sh $start_hour $interval 

work()
{
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
      sample_output="${tfrecords_output_dir}/$ts_hour"
      sample_temp="${tfrecords_job_temp_dir}/$ts_hour"
      ## not filter 
      time gen_tfrecords ${sample_input} ${sample_output}  ${sample_temp} ${py3_env_file} none 0 0 0 &
      ## filter by feature index
      # time gen_tfrecords ${sample_input} ${sample_output}  ${sample_temp} ${py3_env_file} ${encoded_feature_index_list} 0 0 0 &
  done
  wait
}

#time gen_records &
#PID_gen_records=$!
#wait $PID_gen_records

hadoop fs ${name_password} -rm -r ${del_tfrecords_output}

# tfrecordss copyToLocal
copy_records()
{
  for ((i=0; i<$interval; ++i))
  do
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d%H"`
    if [[ -d ${local_tfrecords_data_dir}/${ts_hour}/tfrecords ]]; then
      echo "local tfrecords:${local_tfrecords_data_dir}/${ts_hour}/tfrecords exist for gen-tfrecords.sh delete it first"
      rm -rf ${local_tfrecords_data_dir}/${ts_hour}/tfrecords
    fi
    mkdir -p ${local_tfrecords_data_dir}/${ts_hour}/tfrecords
    for (( j=0; j<10; ++j ))
    do
      time hadoop fs ${name_password} -copyToLocal ${tfrecords_output_dir}/${ts_hour}/tfrecord.*${j} ${local_tfrecords_data_dir}/${ts_hour}/tfrecords &
    done
    # fi
  done
  wait
}

time copy_records &
PID_copy_records=$!
wait $PID_copy_records
show_err $? "copy tfrecords"


for ((i=0; i<$interval; ++i))
do
  ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d%H"`
  python ./scripts/count-tfrecords.py ${local_tfrecords_data_dir}/${ts_hour}/tfrecords &
done
wait

}

time work 
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"end....gen tfrecords"
