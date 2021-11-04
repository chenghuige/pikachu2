
start_hour=$1
interval=$2

source ./function.sh $start_hour $interval 


input_dir=${stats_feature_output_dir}
if (($is_interval_dirs == 1))
then
input=$(interval_dirs $input_dir $start_hour $interval)
else
input=$input_dir
fi
output=$input

# # dedup ---------------------------  not dedup
# if (($interval != 1)) && (($dedup == 1))
# then
# output=${dedup_features_output}
# fi

# # filter ---------------------------  not filter
# if (($interval != '1')) && (($filter == 1))
# then
# output=${filter_features_output}
# fi

# stats ---------------------------
input=${output}
output=${stats_features_output}

time stats_features ${input} ${output} &
wait 

# hadoop fs $name_password -test -e ${stats_features_output}/feature.txt
# if [ $? -ne 0 ];then
#   echo '----------hadoop copy feature.txt'
#   time hadoop fs ${name_password} -cat ${stats_features_output}/feature/* | hadoop fs -put - ${stats_features_output}/feature.txt &
#   echo '----------hadoop copy field.txt'
#   time hadoop fs ${name_password} -cat ${stats_features_output}/field/* | hadoop fs -put - ${stats_features_output}/field.txt &
#   wait
# fi

hadoop fs ${name_password} -rm -r ${del_stats_features_output}
echo '----------------stats_feature end' 

