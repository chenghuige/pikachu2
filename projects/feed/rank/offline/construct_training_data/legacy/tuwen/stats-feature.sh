
start_hour2=$1
if [ $# == 1 ] ; then
interval2=1
else
interval2=$2
fi

if((`expr length $1`!=`expr length 2019100100`))
then
echo "bad input start hour "$start_hour2
exit 1
fi

source  ./function.sh $start_hour2 1

hadoop fs ${name_password} -rm -r ${del_stats_feature_output} 

input=$gen_feature_output
output=$stats_feature_output
time stats_feature $input $output 
wait


