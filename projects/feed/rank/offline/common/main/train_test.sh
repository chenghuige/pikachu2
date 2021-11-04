#!/usr/bin/
hour=`date +\%Y\%m\%d\%H`
month=`date +\%Y\%m`
#start_hour=`date +\%Y\%m\%d\%H`
start_hour=`date -d "-3 hours" "+%Y%m%d%H"`
interval=24
model_name="dlrm-att-18"
mark=$1
abtestid=$2

if (($#>2)) 
then
start_hour=$3
fi

root='/search/odin/libowei/rank'
mkdir -p $root/log/run

echo "cur_hour: "${hour}
echo "start_hour: "${start_hour}
echo "model_name: "${model_name}

sgsapp_data_dir="/search/odin/publicData/CloudS/libowei/rank2/sgsapp/data/${mark}_hour_sgsapp_v1/tfrecords/${start_hour}"
shida_data_dir="/search/odin/publicData/CloudS/libowei/rank2/shida/data/${mark}_hour_shida_v1/tfrecords/${start_hour}"
newmse_data_dir="/search/odin/publicData/CloudS/libowei/rank2/newmse/data/${mark}_hour_newmse_v1/tfrecords/${start_hour}"

run_num=1
while :
do
    echo "***" ${run_num}
    files_count=`ls ${sgsapp_data_dir} ${shida_data_dir} ${newmse_data_dir} | wc -l`
    if [[ "${files_count}" -gt 150 ]]; then
        echo ${sgsapp_data_dir} "Ready!"
        break
    else
        if [ ${run_num} -eq 720 ];then
            echo ${sgsapp_data_dir} "Timeout..."
            msg="${sgsapp_data_dir} wait Timeout!"
            #sh ../../common/send_xiaop.sh "${msg}"
            exit 0
        fi
        date
        echo ${sgsapp_data_dir} "Waiting... ${run_num}"
        sleep 10
        run_num=$(( $run_num+1 ))
    fi
done

echo `date +"%Y%m%d %H:%M:%S"`$'\t'"${start_hour} train model..."
#cd ../../train_and_predict_model/${mark}
#sh -x run.sh ${start_hour} ${abtestid} >> $root/log/run/train_model_${mark}_${model_name}_${start_hour}_${abtestid}.log 2>&1 

#tmpwatch -afv 6 "${root}/log/run"
