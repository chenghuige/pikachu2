
start_hour2=$1
if [ $# == 1 ] ; then
interval2=1
else
interval2=$2
fi

if ((`expr length $1`!=`expr length 2019100100`))
then
echo "bad input start hour "$start_hour2
exit 1
fi

source  ./function.sh $start_hour2 1

deal()
{
    hadoop fs $name_password -mkdir -p ${gen_feature_exist_flag}
    hadoop fs ${name_password} -rm -r ${del_gen_feature_output} 
    hadoop fs ${name_password} -rm -r ${del_eval_output} 
    hadoop fs ${name_password} -rm -r ${del_stats_feature_output} 

    input=$sample_input_dir
    output=$gen_feature_output
    time gen_feature $input $output

    if (($del_inter_result==1))
    then
      echo '--------------deal inter gen feature dirs:'
      hadoop fs ${name_password} -rm -r ${inter_gen_feature_dirs}
    fi

    #input=$gen_feature_output 
    #output=$eval_output
    #time prepare_eval $input $output &  

    input=$gen_feature_output
    output=$stats_feature_output
    time stats_feature $input $output 
    #wait
}

# # construct_feature... -------------------------------------------------------------
gen_features()
{
for (( i=0; i<${interval2}; ++i ))
do
    ts_hour=`date -d"${start_hour2:0:8} ${start_hour2:8:10} -${i}hours" +"%Y%m%d%H"`
    source  ./function.sh $ts_hour 1
    hadoop fs $name_password -test -e ${gen_feature_exist_flag}
    if [ $? -ne 0 ];then
        echo "----------Deal ${ts_hour}"
        deal $ts_hour &
    else
        echo "----------Undo for ${ts_hour} as ${gen_feature_exist_flag} exists"
    fi
done
wait
echo "done gen_feature"
}

time gen_features &
wait

echo "gen_features done for ${start_hour2}_${interval2}"

