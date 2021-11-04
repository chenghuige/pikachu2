#!/usr/bin/
hour=`date +\%Y\%m\%d\%H`
month=`date +\%Y\%m`
#start_hour=`date +\%Y\%m\%d\%H`
start_hour=`date -d "-1 hours" "+%Y%m%d%H"`
interval=24
retrain_flag="yes"
model_name="sgsapp_wdfield_interest_hour"
root='/home/gezi/tmp/rank'
mkdir -p $root/log/run

echo "cur_hour: "${hour}
echo "start_hour: "${start_hour}
echo "model_name: "${model_name}
echo `date +"%Y%m%d %H:%M:%S"`$'\t'"${start_hour} extract feature data..."
cd ../../construct_training_data/tuwen.v2
sh -x run.sh ${start_hour} ${interval} > $root/log/run/construct_feature_${model_name}_${start_hour}_${interval}.log 2>&1

#echo `date +"%Y%m%d %H:%M:%S"`$'\t'"${start_hour} train wide_deep_model..."
#cd ../../train_and_predict_model/tuwen
#sh -x run.sh ${start_hour} ${interval} ${model_name} >> $root/log/run/train_model_${model_name}_${start_hour}_${interval}.log 2>&1 &
#cd ../tuwen2
#sh -x run.sh ${start_hour} ${interval} ${model_name} >> $root/log/run2/train_model_${model_name}_${start_hour}_${interval}.log 2>&1 &
#
#wait
#echo `date +"%Y%m%d %H:%M:%S"`$'\t'"${start_hour} finished"
#

tmpwatch -afv 6 "${root}/log/run"
