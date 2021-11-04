source /root/.bashrc
chgpy2
input_dir=hdfs://GodSonNameNode2/user/traffic_dm/fujinbing/real_show_feature_new
#20191018/2019101815/10/ sleep 120
start_day=`date +\%Y\%m\%d`
start_hour=`date +\%Y\%m\%d\%H`
now_minute=`date +%M`
now_minute=$((10#$now_minute)) # for 08 
start_minute=$(($now_minute / 10 * 10))

if (($start_minute==0))
then
start_minute=00
fi

input_dir=$input_dir/$start_day/$start_hour/

sleep 63

alarm()
{
    msg=`echo -e "${start_hour}\n"`
    msg=`echo -e "$msg\n"`
    for args in $@
    do
        msg="$msg $args"
    done

    sh ./send_xiaop.sh "${msg}"
}


run_tuwen()
{
 log_file=/home/gezi/tmp/rank/log/tuwen_hour.txt
 ty=0
 result_tuwen=`time spark-submit --queue=feedflow_online ./scripts/abinfos.py $baseids $testids $ty $input_dir &`
 echo -e "${start_hour}\t${result_tuwen}" | sed ":a;N;s/\\n/ /g;ta" >> $log_file
 alarm "tuwen $result_tuwen"
}

run_video()
{
  log_file=/home/gezi/tmp/rank/log/video_hour.txt
  ty=1
  result_video=`time spark-submit --queue=feedflow_online ./scripts/abinfos.py $baseids $testids $ty $input_dir &`
  echo -e "${start_hour}\t${result_video}" | sed ":a;N;s/\\n/ /g;ta" >> $log_file
  alarm "video $result_video"
}

baseids='4,5,6'
testids=15
run_tuwen &

testids=8
run_video 

wait
echo "done"

