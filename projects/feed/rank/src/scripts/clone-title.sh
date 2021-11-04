interval=$1
start_hour=2019122813
for ((i=0; i<$interval; i+=1))
do
  ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} +${i}hours" +"%Y%m%d%H"`
  echo $ts_hour
  nc clone-title.py /search/odin/publicData/CloudS/baili/rank/sgsapp/data/tuwen_hour_sgsapp_v1/tfrecords/$ts_hour
  nc clone-title.py /search/odin/publicData/CloudS/baili/rank/newmse/data/tuwen_hour_newmse_v1/tfrecords/$ts_hour
  nc clone-title.py /search/odin/publicData/CloudS/baili/rank/shida/data/tuwen_hour_shida_v1/tfrecords/$ts_hour
done
