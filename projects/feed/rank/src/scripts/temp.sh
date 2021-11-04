mark=$1
start_hour=$2
interval=$3

root_dir="/home/gezi/tmp/rank/data/${mark}_hour_sgsapp_v2"
records_dir="${root_dir}/tfrecords"
info_dir="${root_dir}/infos/15"
model_dir="${root_dir}/models/15"
for ((i=0; i<$interval; i+=6))
do
    echo $i
    for ((j=0; j<6; j++))
    do 
      k1=$((i + j))
      k2=$((i + j + 1))
      #echo $k1 $k2
      ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${k1}hours" +"%Y%m%d%H"`
      ts_hour2=`date -d"${start_hour:0:8} ${start_hour:8:10} -${k2}hours" +"%Y%m%d%H"`
      echo $ts_hour $ts_hour2
      mkdir -p ${info_dir}/${ts_hour}
      #nc infer.py  ${records_dir}/${ts_hour} ${model_dir}/${ts_hour2} > ${info_dir}/${ts_hour}/scores &
    done
    wait 
done

