start_hour=$1
interval=$2

deal()
{
  sh gen-features.sh $1 
  sh gen-tfrecords.sh $1 
}

for ((i=interval; i>0;--i))
do
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d%H"`
    #deal $ts_hour &
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i+1}hours" +"%Y%m%d%H"`
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i+2}hours" +"%Y%m%d%H"`
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i+3}hours" +"%Y%m%d%H"`
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i+4}hours" +"%Y%m%d%H"`
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i+5}hours" +"%Y%m%d%H"`
    echo $ts_hour $i
done
