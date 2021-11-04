start_hour=$1
interval=$2

day=0
interval=$(($interval-1))
for ((i=interval; i>=0; i--))
do
    day=$(($day+1))
    hours=$(($i*24))
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${hours}hours" +"%Y%m%d%H"`
    echo "----------------------------------------day ${day}, with valid hour ${ts_hour}"
    sh ./train/v12/all-video.sh $ts_hour 24 $ts_hour \
        --use_step_file \
	--del_inter_model=1 \
        --save_interval_epochs=-1 \
        --save_interval_steps=10000000000 \
       $*
done
