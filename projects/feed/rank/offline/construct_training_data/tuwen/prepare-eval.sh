
## TODO add debug mode with full stats of each step

start_hour=$1
interval=1

source ./function.sh $start_hour $interval 

local_tfrecords_data_dir=${v2_local_tfrecords_data_dir}

if (($start_hour2 != 0)) 
then
# local_tfrecords_data_dir=${local_tfrecords_data_dir}/${start_hour}_${interval}-${start_hour2}_${interval2}
local_tfrecords_data_dir=${local_tfrecords_data_dir}/${start_hour}
fi

input_dir=${deal_eval_input}

mkdir -p ${local_tfrecords_data_dir}/online_valid

echo "copy from ${input_dir} to ${local_tfrecords_data_dir}/online_valid start"
for (( i=0; i<10; ++i ))
do
  time hadoop fs ${name_password} -copyToLocal ${input_dir}/part-*$i ${local_tfrecords_data_dir}/online_valid &
done

echo "copy from ${input_dir} to ${local_tfrecords_data_dir}/online_valid finished"
wait



