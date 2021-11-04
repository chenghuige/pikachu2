
start_hour=$1
interval=$2

source ./config.sh $start_hour $interval

prefix=${user}-${start_hour}-${model_name}

alarm()
{
    msg=`echo -e "${model_name}_${start_hour}_${interval}\nextract_feature"`
    for args in $@
    do 
        msg=`echo -e "$msg\n$args"`
    done
    sh ./../../common/send_xiaop.sh "$msg"
}
# ok
show_err()
{
if [[ $1 != 0 ]];then
    msg="Error occured when get "$2
    echo $msg
    alarm $msg
    exit -1
fi
}

count()
{
    output_=$1
    echo 'for count: '$output_
    # spark-submit --queue=$queuename --name=$prefix-count ./scripts/count.py $output_ &
}

interval_dirs()
{
    input_dir=$1
    start_hour=$2
    interval=$3
    input=''
    begin_num=0
    if (($#>3))
    then
    begin_num=$4
    fi

    for ((i=${begin_num}; i<$interval; ++i))
    do
        ts_day=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d"`
        ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d%H"`
        sample_input="$input_dir/$ts_hour/"
        # echo 'sample input: '$sample_input
        hadoop fs $name_password -test -e $sample_input
        if [ $? -eq 0 ];then
            input="${sample_input},${input}"
        # echo 'input: '$input
        fi
    done
    if [ -n "$input" ]; then
        input=${input:0:-1}
    fi
    
    echo $input
}

#--------------------------------------------------------------------

gen_feature()
{
    name="gen_feature"
    echo '------------------: '$name
    echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'$name"...." 

    input=$1
    output_feature=$2
    output_utf8=$gen_feature_output_utf8
    output_dedup=$gen_feature_output_dedup
    output_filter=$gen_feature_output_filter
    output_join=$gen_feature_output_join
    echo "input:" $input
    echo "output:" $output_feature
    echo "model_name:" $model_name
    
    hadoop fs $name_password -test -e $output_feature
    if [ $? -eq 0 ];then
    #   hadoop fs $name_password -rm -r $output_feature
        echo "[${output}] already exist do nothing for function gen_feature" 
        return 0
    fi

    ts_day=`date -d"${start_hour:0:8} ${start_hour:8:10}" +"%Y%m%d"`
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10}" +"%Y%m%d%H"`
    input="$input/$ts_day/$ts_hour/*/"
    # input="$input/$ts_day/$ts_hour/00/"

    hadoop fs $name_password -test -e $input
    if [ $? -eq 0 ];then
        base_feature_input=" $input"
    fi

    output=$input
    #--------------------------------------prepare before gen feature
    #------------------------------.utf8
    output=$output_utf8
    hadoop fs $name_password -test -e $output_utf8
    if [ $? -ne 0 ];then
        echo '------convert from gbk to utf8'
        echo 'input:'$input
        echo 'output:'$output
        # time sh ./gbk2utf8-spark.sh $input $output
        # time sh ./gbk2utf8-hadoop.sh $input $output
        time spark-submit \
            --name $prefix-gbk2utf8 \
            --class com.appfeed.sogo.to_utf8 \
            --master yarn --num-executors 500 \
            --driver-memory 3g \
            --executor-cores 4 \
            --executor-memory 3g \
            --conf spark.hive.mapred.supports.subdirectories=true \
            --conf spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive=true \
            --conf spark.ui.showConsoleProgress=true \
            --conf spark.yarn.queue=root.$queuename \
            --conf spark.driver.maxResultSize=3g \
            --conf spark.dynamicAllocation.enabled=true \
            --conf spark.port.maxRetries=100 \
            --conf spark.app.name=$prefix-gbk2utf8 \
            gbk2utf8.jar \
                $input \
                $output \
                500
        count $output
    fi

    #------------------------------.dedup
    input=$output
    output=$output_dedup
    hadoop fs $name_password -test -e $output_dedup
    if [ $? -ne 0 ];then
        echo '------dedup ori'
        echo 'input:'$input 
        echo 'output:'$output
        time spark-submit --name=$prefix-dedup-ori --queue=$queuename --py-files=./config.py \
            ./scripts/dedup-ori.py $input $output 
        count $output
    fi

    #------------------------------.filter
    input=$output
    output=$output_filter
    hadoop fs $name_password -test -e $output_filter
    if [ $? -ne 0 ];then
        echo '------filter ori'
        echo 'input:'$input 
        echo 'output:'$output
        echo 'mdocid_output:'${gen_feature_output_mdocid}
        echo 'read_mid_history_interval:' ${read_mid_history_interval}
        start_num=1 #not include cur hour
        history_mdocid=$(interval_dirs "${gen_feature_output_mdocid_dir}" "$start_hour" "${read_mid_history_interval}" "${start_num}")
        if [ -z "${history_mdocid}" ]; then
            history_mdocid='None'
        fi
        echo 'history_mdocid_input:'${history_mdocid}
        time spark-submit --name=$prefix-filter-ori --queue=$queuename --py-files=./config.py \
            ./scripts/filter-ori.py $input $output ${gen_feature_output_mdocid} ${history_mdocid}
        count $output
    fi

    #------------------------------.join
    if (($join == 1))
    then
        input=$output
        output=$output_join
        hadoop fs $name_password -test -e $output_join
        if [ $? -ne 0 ];then
            echo '------join show for lr score, this is optional and now used only for comparing with online prediction'
            echo 'input:'$input 
            echo 'output:'$output
            input2="${input_show_dir}/${ts_day}/${ts_hour}/"
            time spark-submit --name=$prefix-join-show --queue=$queuename --py-files=./config.py \
                ./scripts/join-show.py $input $input2 $output 
            count $output
        fi
    fi

     #--------------------------------------gen feature
    echo '------gen feature'
    input=$output
    output=$output_feature
    echo 'input:'$input
    echo 'output:'$output
    time spark-submit --name=$prefix-gen-feature --queue=$queuename --py-files=./config.py \
        ./scripts/gen-feature.py $input $output $model_name 

    show_err $? $name
    count $output

}

prepare_eval()
{
    name="eval"
    echo '------------------: '$name
    echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'$name"...." 
    input=$1
    output=$2

    hadoop fs $name_password -test -e $output
    if [ $? -eq 0 ];then
    #   hadoop fs $name_password -rm -r $output_feature
        echo "[${output}] already exist do nothing for function prpeapre_eval" 
        return 0
    fi

    echo "input:" $input
    echo "output:" $output 

    spark-submit --name=$prefix-prepare-eval --queue=$queuename --py-files=./config.py \
        ./scripts/prepare-eval.py $input $output 
    
    show_err $? $name
    echo 'output: '$output
}

stats_feature()
{
    name="stats_feature"
    echo '------------------: '$name
    echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'$name"...." 
    input=$1
    output=$2

    hadoop fs $name_password -test -e $output
    if [ $? -eq 0 ];then
    #   hadoop fs $name_password -rm -r $output_feature
        echo "[${output}] already exist do nothing for function stats_feature" 
        return 0
    fi

    echo "input:" $input
    echo "output:" $output 

    spark-submit --name=$prefix-stats-feature --queue=$queuename --py-files=./config.py \
        ./scripts/stats-feature.py $input $output &

    spark-submit --name=$prefix-stats-field --queue=$queuename --py-files=./config.py \
        ./scripts/stats-field.py $input $output &

    wait
    
    show_err $? $name
    count "${output}/feature_ori"
    count "${output}/feature"
    count "${output}/field"
}

stats_field()
{
    name="stats_field"
    echo '------------------: '$name
    echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'$name"...." 
    input=$1
    output=$2

    hadoop fs $name_password -test -e $output
    if [ $? -eq 0 ];then
    #   hadoop fs $name_password -rm -r $output_feature
        echo "[${output}] already exist do nothing for function stats_field" 
        return 0
    fi

    echo "input:" $input
    echo "output:" $output 

    spark-submit --name=$prefix-stats-field --queue=$queuename --py-files=./config.py \
        ./scripts/stats-field.py $input $output &

    wait
    
    show_err $? $name
    count "${output}/field"
}


#-------------deprecated not use
dedup_features()
{
    name="dedup_features"
    echo '------------------: '$name
    echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'$name"...." 
    input=$1
    output=$2
    hadoop fs $name_password -test -e $output
   if [ $? -eq 0 ];then
    #   hadoop fs $name_password -rm -r $output_feature
        echo "[${output}] already exist do nothing for function dedup_features" 
        return 0
    fi

    echo "input:" $input
    echo "output:" $output

    spark-submit --name=$prefix-dedupe-features --queue=$queuename --py-files=./config.py \
        ./scripts/dedup-features.py $input $output 

    show_err $? $name
    count $output
}

#-------------deprecated not use, should also be named as filter_features
filter_features()
{
    name="filter_features"
    echo '------------------: '$name
    echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'$name"...." 
    input=$1
    output=$2
    hadoop fs $name_password -test -e $output
    if [ $? -eq 0 ];then
    #   hadoop fs $name_password -rm -r $output_feature
        echo "[${output}] already exist do nothing for function filter_feature" 
        return 0
    fi

    echo "input:" $input
    echo "output:" $output

    spark-submit --name=$prefix-filter-features --queue=$queuename --py-files=./config.py \
        ./scripts/filter-features.py $input $output $interval 

    show_err $? $name
    count $output
}

stats_features()
{
    name="stats_features"
    echo '------------------: '$name
    echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'$name"...." 
    input=$1
    output=$2

    hadoop fs $name_password -test -e $output
    if [ $? -eq 0 ];then
    #   hadoop fs $name_password -rm -r $output_feature
        echo "[${output}] already exist do nothing for function stats_features" 
        return 0
    fi

    echo "input:" $input
    echo "output:" $output 

    spark-submit --name=$prefix-stats-features --queue=$queuename --py-files=./config.py \
        ./scripts/stats-features.py $input $output &

    spark-submit --name=$prefix-stats-fields --queue=$queuename --py-files=./config.py \
        ./scripts/stats-fields.py $input $output &

    wait

    show_err $? $name
    count "${output}/feature_ori"
    count "${output}/feature"
    count "${output}/field"
}

gen_tfrecords()
{
    name="gen_tfrecords"
    echo '------------------: '$name
    echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'$name"...." 
    # for debug
    input=$1

    if (($tfrecords_debug==1))
    then
      input=$1"/part-00000*"
      all_parts=1
    fi
    
    output=$2
    reducer_temp=$3
    py3env=$4
    
    hadoop fs $name_password -test -e $output
    if [ $? -eq 0 ];then
    #   hadoop fs $name_password -rm -r $output_feature
        echo "[${output}] already exist do nothing for function gen_tfrecords" 
        return 0
    else
        echo "hadoop fs $name_password mkdir -p $output"
        hadoop fs $name_password -mkdir -p $output
    fi

    hadoop fs $name_password -test -e $reducer_temp
    if [ $? -eq 0 ];then
      hadoop fs $name_password -rm -r $reducer_temp
    fi

    echo "input:" $input
    echo "output:" $output
    echo "reducer_out(temp):" "$reducer_temp"
    echo "cacheArchive py3env:" "$py3env"
    echo "has_emb:" "$6"
    echo "use_emb:" "$7"
    echo "portrait_emb_dim:" "$8"
    echo "queuename:" $queuename

    feature_index=$5
    script=gen-tfrecords.py

    count $input

    # -D mapreduce.job.queuename=feedflow_online \

    ## need below if you need filter feature by dict TODO
    # -cacheFile $feature_index#feature_index \

    hadoop org.apache.hadoop.streaming.HadoopStreaming \
    ${name_password} \
    -D mapred.job.name="${prefix}-gen-tfrecords" \
    -D mapreduce.job.queuename=$queuename \
    -D mapreduce.map.memory.mb=1024 \
    -D mapreduce.reduce.memory.mb=4096 \
    -D mapreduce.input.fileinputformat.split.minsize=5000000000 \
    -input $input \
    -output $reducer_temp \
    -cacheArchive $py3env#py3 \
    -file ./scripts/$script \
    -file ./config.py \
    -numReduceTasks $all_parts \
    -mapper "py3/libowei/bin/python3 $script mapper" \
    -reducer "py3/libowei/bin/python3 $script reducer $output $feature_index $6 $7 $8" 

    hadoop fs ${name_password} -rm -r $reducer_temp

    show_err $? ${name}
    echo 'output: '$output
    # count $output
}

