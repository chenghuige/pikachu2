#!/usr/bin/
hour=`date +\%Y\%m\%d\%H`
month=`date +\%Y\%m`
#start_hour=`date +\%Y\%m\%d\%H`
start_hour=`date -d "-1 hours" "+%Y%m%d%H"`
interval=24
retrain_flag="yes"
model_name="sgsapp_wdfield_interest_hour"
abtestid=$1
root='/home/gezi/tmp/rank'
mkdir -p $root/log/run

echo "cur_hour: "${hour}
echo "start_hour: "${start_hour}
echo "model_name: "${model_name}
echo `date +"%Y%m%d %H:%M:%S"`$'\t'"${start_hour} extract feature data..."
cd ../../construct_training_data/tuwen
sh -x run.sh ${start_hour} ${interval} > $root/log/run/construct_feature_tuwen_${model_name}_${start_hour}_${abtestid}.log 2>&1

pushd .
cd /search/odin/publicData/CloudS/chenghuige
cd /search
cd  /search/odin/publicData/CloudS/chenghuige
popd


echo `date +"%Y%m%d %H:%M:%S"`$'\t'"${start_hour} train model..."
cd ../../train_and_predict_model/tuwen
abtestid=15
sh -x run.sh ${start_hour} ${abtestid} >> $root/log/run/train_model_tuwen_${model_name}_${start_hour}_${abtestid}.log 2>&1 
# wait
# echo `date +"%Y%m%d %H:%M:%S"`$'\t'"${start_hour} finished"

# temply also use 16
echo `date +"%Y%m%d %H:%M:%S"`$'\t'"${start_hour} train model..."
cd ../../train_and_predict_model/tuwen
abtestid=16
sh -x run.sh ${start_hour} ${abtestid} >> $root/log/run/train_model_tuwen_${model_name}_${start_hour}_${abtestid}.log 2>&1 
# wait
# echo `date +"%Y%m%d %H:%M:%S"`$'\t'"${start_hour} finished"

tmpwatch -afv 6 "${root}/log/run"
