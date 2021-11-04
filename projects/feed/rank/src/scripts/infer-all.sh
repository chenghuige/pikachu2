mark=$1
start_hour=$2
interval=$3

root_dir="/search/odin/publicData/CloudS/baili/new_rank/data/${mark}_hour_sgsapp_v1"
records_dir="${root_dir}/tfrecords"
info_dir="/search/odin/publicData/CloudS/rank/infos/${mark}/16.new"
model_dir="${root_dir}/models/16"
span=$4
for ((i=0; i<$interval; i+=$span))
do
    echo $i
    for ((j=0; j<$span; j++))
    do 
      k1=$((i + j))
      k2=$((i + j + 2))
      #echo $k1 $k2
      ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${k1}hours" +"%Y%m%d%H"`
      ts_hour2=`date -d"${start_hour:0:8} ${start_hour:8:10} -${k2}hours" +"%Y%m%d%H"`
      mkdir -p ${info_dir}/${ts_hour}
      infer.py ${records_dir}/${ts_hour} ${model_dir}/${ts_hour2} --ofile=${info_dir}/${ts_hour}/valid.csv &
    done
    wait 
done

#for ((i=0; i<$interval; i+=1))
#do
#  k1=$((i))
#  k2=$((i + 1))
#  #echo $k1 $k2
#  ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${k1}hours" +"%Y%m%d%H"`
#  ts_hour2=`date -d"${start_hour:0:8} ${start_hour:8:10} -${k2}hours" +"%Y%m%d%H"`
#  echo $ts_hour $ts_hour2
#  mkdir -p ${info_dir}/${ts_hour}
#  infer.py ${records_dir}/${ts_hour} ${model_dir}/${ts_hour2} --ofile=${info_dir}/${ts_hour}/valid.csv
#done
#
