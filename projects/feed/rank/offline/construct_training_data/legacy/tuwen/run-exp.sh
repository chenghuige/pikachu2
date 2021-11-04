
# see README.md for usage 

start_hour_=$1

interval_=1
if (($#>1))
then
interval_=$2  # do not use interval as it is in config.sh will be modified when source cofig.sh
fi 

valid_interval=1
if (($#>2)) 
then
valid_interval=$3
fi

span=0
if (($#>3))
then
span=$4
fi

echo "---begin run exp start:$start_hour_ train_interval:$interval_ valid_interval:$valid_interval span:$span"

source ./config.sh $start_hour_ $interval_

ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -$((span+valid_interval))hours" +"%Y%m%d%H"`

#------gen features 
x=$start_hour_
y=$((interval_+valid_interval+span))
echo "-----gen feature for ${x}_${y} begin"
source ./config.sh $x $y
time sh ./gen-features.sh $x $y
echo "-----gen feature for ${x}_${y} done"

x=$ts_hour
y=$interval_

time sh stats-features.sh $x $y
echo "----------------stats_features end"

#-----deal for train
source ./config.sh $x $y
echo "-----For training deal feature for ${x}_${y} begin"
time sh ./deal-features.sh $ts_hour $interval 0 1 & # train but not stats (interval != 0) , not prepare eval (start hour == 0)
PID_train=$!

#-------deal valid
x=$start_hour_
y=$valid_interval

source ./config.sh $x $y
echo "-----For valid deal feature for ${x}_${y} with feature index ${ts_hour}_${interval_} begin"
time sh ./deal-features.sh $x $y $ts_hour $interval_ &
PID_valid=$!

wait $PID_train
echo "-----deal train feature for ${x}_${y} done"
wait $PID_valid
echo "-----deal valid feature for ${x}_${y} with feature index ${ts_hour}_${interval_} done"

echo "done run exp start:$start_hour_ train_interval:$interval_ valid_interval:$valid_interval span:$span"
