start_hour=$1
interval=$2

for ((i=interval-1; i>=0; --i))
do
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d%H"`
    sh ./train/v12/all-video.sh $ts_hour 1 $ts_hour \
        --use_step_file \
        --del_inter_model=1 \
        --save_interval_epochs=-1 \
        --save_interval_steps=100000000000 \
       $*
    echo $ts_hour
done
