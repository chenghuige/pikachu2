start_hour=$1
if [ $# == 1 ] ; then
interval=1
else
interval=$2
fi

if((`expr length $1`!=`expr length 2019100100`))
then
echo "bad input start hour "$start_hour
exit 1
fi

source ./function.sh $start_hour $interval 

sh gen-features.sh ${start_hour} $interval

sh gen-tfrecords.sh ${start_hour} $interval

sh copy-tfrecords.sh ${start_hour} $interval

echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"end....deal features"



