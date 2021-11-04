mark=$1
start_hour=$2
interval=$3

root_dir="/home/gezi/tmp/rank/data_newmse/${mark}_hour_newmse_v1"
records_dir="${root_dir}/tfrecords"
info_dir="/search/odin/publicData/CloudS/rank/infos/${mark}/common"
span=$4
for ((i=0; i<$interval; i+=$span))
do
    echo $i
    for ((j=0; j<$span; j++))
    do 
      k=$((i + j))
      ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${k}hours" +"%Y%m%d%H"`
      echo $ts_hour 
      mkdir -p ${info_dir}/${ts_hour}
      echo ${records_dir}/${ts_hour}
      CUDA_VISIBLE_DEVICES=-1 loop.py ${records_dir}/${ts_hour} --ofile=${info_dir}/${ts_hour}/infos_newmse.csv --title &
    done
    wait 
done

