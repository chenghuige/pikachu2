#!/usr/bin/
start_hour=$1
interval=$2

source ./config.sh $start_hour $interval

# just for safe $inverval actually only latest 1 will run if running each hour ontime
# time sh ./gen-features.sh $start_hour $interval &
time sh ./gen-features.sh $start_hour 3 &
wait
echo "gen feature for 1 hour done for ${start_hour}_${interval}" 

# TODO actually now is 1 hour also for online learning, interval not used in deal-features
time sh ./deal-features.sh $start_hour $interval &
PID_train=$!

wait $PID_train
echo "deal features for ${interval} hours done for ${start_hour}_${interval}"
