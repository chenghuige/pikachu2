start_hour=$1
interval=1
if (($#>1)) 
then
interval=$2
fi

source ./function.sh $start_hour $interval 

local_tfrecords_data_dir_=${local_tfrecords_data_dir}
if (($#>2)) 
then
local_tfrecords_data_dir_=$3
fi


# tfrecordss copyToLocal
copy_records()
{
  for ((i=0; i<$interval; ++i))
  do
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d%H"`
    local_tfrecords=${local_tfrecords_data_dir_}/tfrecords/${ts_hour}
    if [[ -d $local_tfrecords ]]; then
      # echo "local tfrecords:${local_tfrecords} exist"
      is_localdir_ok ${local_tfrecords}
      if (($?==1))
      then
        echo "ignore as dir is ok ${local_tfrecords}"
        continue
      else
         echo "delete and recopy as dir not valid ${local_tfrecords}"
         sudo rm -rf ${local_tfrecords}
      fi
    fi
    mkdir -p ${local_tfrecords}
    touch ${local_tfrecords}.lock
    for (( j=0; j<10; ++j ))
    do
      time hadoop fs ${name_password} -copyToLocal ${tfrecords_output_dir}/${ts_hour}/tfrecord.*${j} ${local_tfrecords} &
    done
    # fi
  done
  wait
  for ((i=0; i<$interval; ++i))
  do
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d%H"`
    local_tfrecords=${local_tfrecords_data_dir_}/tfrecords/${ts_hour}
    sudo rm -rf ${local_tfrecords}.lock
  done
  
}

time copy_records &
PID_copy_records=$!
wait $PID_copy_records
show_err $? "copy tfrecords"


for ((i=0; i<$interval; ++i))
do
  ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d%H"`
  local_tfrecords=${local_tfrecords_data_dir_}/tfrecords/${ts_hour}
  # if [[ -d $local_tfrecords/num_records.txt ]]; then
  #     echo "local tfrecords:${local_tfrecords}/num_records.txt exist for gen-tfrecords.sh ignore it"
  #     # sudo rm -rf ${local_tfrecords}
  #     continue
  # fi
  python ./scripts/count-tfrecords.py ${local_tfrecords} &
done
wait

echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"end....gen tfrecords"
