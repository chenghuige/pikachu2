is_hdfsdir_ok()
{
    input_=$1
    num_files=`hadoop fs -count $input_| awk '{print $2}'`
    size_=`hadoop fs -du -s $input_| awk '{print $1}'`
    if (($num_files >= 50 && $size_ > 100000))
    then
        return 1
    else
        return 0
    fi
}

is_hdfsdir_ok /user/traffic_dm/chg/rank/video_hour_sgsapp_v2/tfrecords/2019120109
echo $?
if (($?==1))
then 
	echo "ok"
else
	echo "wrong"
fi

# is_localdir_ok()
# {
#     input_=$1
#     num_files=`ls -l $input_ | grep "^-" | wc -l`
#     size_=`du -l $input_| awk '{print $1}'`
#     if (($num_files >= 50 && $size_ > 100000))
#     then
#         return 1
#     else
#         return 0
#     fi
# }

is_localdir_ok()
{
    input_=$1
    num_files=`ls -l $input_ | grep "^-" | wc -l`
    # size_=`du $input_| awk '{print $1}'`
    # if (($num_files >= $all_parts && $size_ > 100000))
    size=`ls $input_ | tail -1 | awk -F'.' '{print $3}'`
    if (($num_files >= 50 && $size != 0))
    then
        return 1
    else
        return 0
    fi
}

is_localdir_ok /search/odin/publicData/CloudS/chenghuige/rank/data/video_hour_sgsapp_v2/tfrecords/2019120109
echo $?