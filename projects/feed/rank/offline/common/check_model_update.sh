#!/usr/bin/
source ./send_sms.sh

update_time_deadline=`date -d "-5 hours" "+%Y-%m-%d %H:%M:%S"`
deadline_time=`date -d "$update_time_deadline" +%s`   
public_dir="/search/odin/public_data/"
public_dir="./"
shida_realtime_wd="/search/fujinbing/predict_model/train_and_predict_model/wide_deep_model/data/shida_online_data/publish"
#out=(${shida_realtime_wd} ${public_dir})
out=(${public_dir})
err_msg="now is "`date `", but "
err_msg1=""
flag=1
for dir in ${out[@]}
do
    echo ${dir}
    for file in ${dir}/*
    do 
        file_update=`stat ${file} | grep Modify | awk '{print $2 " " $3}' | awk -F "." '{print $1}'`
        file_update_time=`date -d "${file_update}" +%s`
        if [ ${file_update_time} -lt ${deadline_time} ];
        then
            err_msg=${err_msg}`ls -l ${file} |awk '{print $9 " update in " $6$7 "th "  $8 ", "}'`
            err_msg1=${err_msg1}`ls -l ${file} |awk '{print $9}'`
            flag=0
        fi
    done
done

if [ $flag -lt 1 ];
then
    echo ${err_msg}
    alarm "${err_msg1}"
    python "send_email.py" "${err_msg}" "some model not update on time!" 1
fi
