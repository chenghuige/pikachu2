#!/usr/bin/
hour=`date +\%Y\%m\%d\%H`
month=`date +\%Y\%m`
#start_hour=`date +\%Y\%m\%d\%H`
start_hour=`date -d "-1 hours" "+%Y%m%d%H"`
#start_hour=`date -d "-3 hours" "+%Y%m%d%H"`  # mkyuwen TODO:
#start_hour="2020070405"  # mkyuwen
interval=24
# model_name="tw-dlrm-att-avgPool-shareKw-1"  # mkyuwen 
mark=$1
abtestid=$2
model_name="dlrm-att-avgPool-shareKw-1"  # mkyuwen 

if (($#>2)) 
then
start_hour=$3
fi

# root='/search/odin/libowei/rank'
root='/search/odin/mkyuwen/rank_onexp_0612'
mkdir -p $root/log/run

echo "cur_hour: "${hour}
echo "start_hour: "${start_hour}
echo "model_name: "${model_name}

# sgsapp_data_dir="/search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/${mark}_hour_sgsapp_v1/tfrecords/${start_hour}/_SUCCESS"
# shida_data_dir="/search/odin/publicData/CloudS/libowei/rank4/shida/data/${mark}_hour_shida_v1/tfrecords/${start_hour}/_SUCCESS"
# newmse_data_dir="/search/odin/publicData/CloudS/libowei/rank4/newmse/data/${mark}_hour_newmse_v1/tfrecords/${start_hour}/_SUCCESS"
sgsapp_data_dir="/search/odin/publicData/CloudS/yuwenmengke/rank_0521/sgsapp/data/${mark}_hour_sgsapp_v1/tfrecords/${start_hour}/_SUCCESS"
shida_data_dir="/search/odin/publicData/CloudS/yuwenmengke/rank_0521/shida/data/${mark}_hour_shida_v1/tfrecords/${start_hour}/_SUCCESS"
newmse_data_dir="/search/odin/publicData/CloudS/yuwenmengke/rank_0521/newmse/data/${mark}_hour_newmse_v1/tfrecords/${start_hour}/_SUCCESS"

run_num=1
while :
do
    echo "***" ${run_num}
    ls ${sgsapp_data_dir} ${shida_data_dir} ${newmse_data_dir} 
    if [ "$?" == "0" ]; then
        echo ${sgsapp_data_dir} "Ready!"
        break
    else
        if [ ${run_num} -eq 72 ];then
            echo ${sgsapp_data_dir} "Timeout..."
            msg="${sgsapp_data_dir} wait Timeout!"
            #sh ../../common/send_xiaop.sh "${msg}"
            exit 0
        fi
        date
        echo ${sgsapp_data_dir} "Waiting... ${run_num}"
        sleep 100
        run_num=$(( $run_num+1 ))
    fi
done

echo `date +"%Y%m%d %H:%M:%S"`$'\t'"${start_hour} train model..."
cd ../../train_and_predict_model/${mark}
sh -x run.sh ${start_hour} ${abtestid} >> $root/log/run/train_model_${mark}_${model_name}_${start_hour}_${abtestid}.log 2>&1 

tmpwatch -afv 6 "${root}/log/run"
