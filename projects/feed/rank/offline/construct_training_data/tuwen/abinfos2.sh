source /root/.bashrc
chgpy2
input_dir=hdfs://GodSonNameNode2/user/traffic_dm/fujinbing/real_show_feature_new
#20191018/2019101815/10/ sleep 120
start_day=`date +\%Y\%m\%d`
start_hour=`date +\%Y\%m\%d\%H`
now_minute=`date +%M`
now_minute=08
now_minute=$((10#$now_minute))
start_minute=$(( $now_minute / 10 * 10))

if (($start_minute==0))
then
start_minute=00
fi

input_dir=$input_dir/$start_day/$start_hour/$start_minute

exit 0
#sleep 63

alarm()
{
    msg=`echo -e "${start_hour}_${start_minute}\n"`
    msg=`echo -e "$msg\n"`
    for args in $@
    do
        msg="$msg $args"
    done

    sh ./send_xiaop.sh "${msg}"
}


baseids='4,5,6'
testids=15
ty=0
result=`time spark-submit ./scripts/abinfos.py $baseids $testids $ty $input_dir`

# result = 'abc'

alarm $result

