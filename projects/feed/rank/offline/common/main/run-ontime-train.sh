#!/usr/bin/
hour=`date +\%Y\%m\%d\%H`
month=`date +\%Y\%m`
#start_hour=`date +\%Y\%m\%d\%H`
#start_hour=`date -d "-1 hours" "+%Y%m%d%H"`
start_hour=$1
interval=24
retrain_flag="yes"
model_name="sgsapp_wdfield_interest_hour"
root='/home/gezi/tmp/rank'
mkdir -p $root/log

echo "cur_hour: "${hour}
echo "start_hour: "${start_hour}
echo "model_name: "${model_name}
echo `date +"%Y%m%d %H:%M:%S"`$'\t'"${start_hour} extract feature data..."
#cd ../../construct_training_data/spark
#sh -x run.sh ${start_hour} ${interval} > $root/log/run_construct_feature_${model_name}_${start_hour}_${interval}.log 2>&1

echo `date +"%Y%m%d %H:%M:%S"`$'\t'"${start_hour} train wide_deep_model..."
cd ../../train_and_predict_model
sh -x run.sh ${start_hour} ${interval} ${model_name} >> $root/log/run_train_model_${model_name}_${start_hour}_${interval}.log 2>&1
echo `date +"%Y%m%d %H:%M:%S"`$'\t'"${start_hour} finished"

tmpwatch -afv 48 "${root}/log/"
