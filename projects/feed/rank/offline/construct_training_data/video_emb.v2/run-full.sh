
# this is offline full , first try to deal 24 hour datas each parallel(if exist will ignore), then merge 24 hours datas for tfrecords

start_hour=$1
interval=$2

# start hour for feature index, 0 means not use, otherwise use pre gen feature index
start_hour2=0
interval2=0
if [ $# == 4 ] ; then
start_hour2=$3
interval2=$4
fi

source ./config.sh $start_hour $interval

time sh ./gen-features.sh $start_hour $interval  
echo "gen feature for ${interval} hours done"

time sh ./deal-features.sh $start_hour $interval $start_hour2 $interval2
echo "deal feature for ${interval} hours done"

