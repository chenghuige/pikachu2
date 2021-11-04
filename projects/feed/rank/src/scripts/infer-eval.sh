mark=$1
start_hour=$2
interval=$3

root_dir_old="/search/odin/publicData/CloudS/baili/new_rank/data/${mark}_hour_sgsapp_v1"
root_dir="/search/odin/publicData/CloudS/baili/rank/sgsapp/data/${mark}_hour_sgsapp_v1"
records_dir="${root_dir}/tfrecords"
info_dir="/search/odin/publicData/CloudS/rank/infos/${mark}/16.new2"
model_dir="${root_dir_old}/models/16"
span=$4

deal()
{
   mkdir -p ${info_dir}/${ts_hour}
   infer.py ${records_dir}/${ts_hour} ${model_dir}/${ts_hour2} --ofile=${info_dir}/${ts_hour}/valid.csv 
   CUDA_VISIBLE_DEVICES=-1 eval-all.py ${info_dir}/${ts_hour}/valid.csv --online_abids=456
}

for ((i=0; i<$interval; i+=$span))
do
    echo $i
    for ((j=0; j<$span; j++))
    do 
      k1=$((i + j))
      k2=$((i + j + 2))
      ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${k1}hours" +"%Y%m%d%H"`
      ts_hour2=`date -d"${start_hour:0:8} ${start_hour:8:10} -${k2}hours" +"%Y%m%d%H"`
      deal &
    done
    wait 
done

