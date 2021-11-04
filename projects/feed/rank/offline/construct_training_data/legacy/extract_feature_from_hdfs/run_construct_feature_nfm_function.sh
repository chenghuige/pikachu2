#!/usr/bin/
#coding=gbk

alarm()
{
    msg=`echo -e "${model_name}_${start_hour}_${interval}\nextract_feature"`
    for args in $@
    do 
        msg=`echo -e "$msg\n$args"`
    done
    sh ./../../common/send_xiaop.sh "${msg}"
}
# ok
show_err()
{
if [[ $1 != 0 ]];then
    msg="Error occured when get "$2
    echo ${msg}
    alarm ${msg}
    exit -1
fi
}
#--------------------------------------------------------------------
filter_bad_case_mid()
{
name="filter_bad_case_mid"
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'${name}"...toCheckTime"

echo "param input:" "$1"
echo "param out_file:" "$2"
echo "param mapper_script:" "$3"
echo "param reducer_script:" "$4"
echo "param model_name:" "$5"
hadoop fs ${name_password} -rm -r $2

hadoop org.apache.hadoop.streaming.HadoopStreaming \
${name_password} \
-D mapred.job.name="extract_feature-${model_name}-filter_bad_case_mid" \
-D mapreduce.input.fileinputformat.split.minsize=5000000000 \
-D mapreduce.map.memory.mb=512 \
-D mapreduce.job.queuename=feedflow_online \
$1 -output $2 -file "./script/$3" -file "./script/$4" -numReduceTasks 32 -mapper "python $3 $5" -reducer "python $4"

show_err $? ${name}
}
filter_bad_case_docid()
{
name="filter_bad_case_docid"
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'${name}"...toCheckTime"

echo "param input:" "$1"
echo "param out_file:" "$2"
echo "param mapper_script:" "$3"
echo "param reducer_script:" "$4"
echo "param model_name:" "$5"
hadoop fs ${name_password} -rm -r $2

hadoop org.apache.hadoop.streaming.HadoopStreaming \
${name_password} \
-D mapred.job.name="extract_feature-${model_name}-filter_bad_case_docid" \
-D mapreduce.input.fileinputformat.split.minsize=5000000000 \
-D mapreduce.map.memory.mb=512 \
-D mapreduce.job.queuename=feedflow_online \
$1 -output $2 -file "./script/$3" -file "./script/$4" -numReduceTasks 32 -mapper "python $3 $5" -reducer "python $4"

show_err $? ${name}
}

combined_feature()
{
name="combined_feature"
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'${name}"...toCheckTime"

echo "param input:" "$1"
echo "param out_file:" "$2"
echo "param mapper_script:" "$3"
echo "param interval json:" "$4"
echo "param model_name:" "$5"
hadoop fs ${name_password} -rm -r $2

ts_day=`date -d"${start_hour:0:8} ${start_hour:8:10}" +"%Y%m%d"`
ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10}" +"%Y%m%d%H"`
sample_input="$sample_input_dir/${ts_day}/${ts_hour}/${product}"
hadoop fs ${name_password} -test -e ${sample_input}
if [ $? -eq 0 ];then
    base_feature_input=" ${sample_input}"
fi

hadoop org.apache.hadoop.streaming.HadoopStreaming \
${name_password} \
-D mapred.job.name="extract_feature-${model_name}-combined_feature" \
-D mapreduce.input.fileinputformat.split.minsize=5000000000 \
-D stream.num.map.output.key.fields=2 \
-D num.key.fields.for.partition=2 \
-D mapreduce.map.memory.mb=4096 \
-D mapreduce.reduce.memory.mb=4096 \
-D mapreduce.job.queuename=feedflow_online \
-partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
-input ${base_feature_input} -output $2 -file $4 -file "./script/$3" -numReduceTasks 64 -mapper "python $3 $5"

show_err $? ${name}
}

filter_train_data()
{
name="filter_train_data"
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'${name}"...toCheckTime"

echo "param input:" "$1"
echo "param out_file:" "$2"
echo "param mapper_script:" "$3"
echo "param filter_mid_data_dir:" "$4"
echo "param filter_mid_data:" "$5"
echo "param model_name:" "$6"
hadoop fs ${name_password} -rm -r $2

hadoop org.apache.hadoop.streaming.HadoopStreaming \
${name_password} \
-D mapred.job.name="extract_feature-${model_name}-filter_train_data" \
-D mapreduce.input.fileinputformat.split.minsize=5000000000 \
-D stream.num.map.output.key.fields=2 \
-D mapreduce.map.memory.mb=4096 \
-D mapreduce.reduce.memory.mb=4096 \
-D mapreduce.job.queuename=feedflow_online \
$1 -output $2 -file "./script/$3" -file "$4/$5" -numReduceTasks 64 -mapper "python $3 $5 $6"

show_err $? ${name}
}


dedup_feature_list_new()
{
name="dedup_feature_list_new"
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'${name}"...toCheckTime"
sample_input_dir=$1
interval=$2
output=$3
product=$4
for (( i=0; i<=${interval}; ++i ))
do
    ts_day=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d"`
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d%H"`
    sample_input="$sample_input_dir/${ts_hour}/"
    hadoop fs ${name_password} -test -e ${sample_input}
    if [ $? -eq 0 ];then
        dedup_input=" ${sample_input}${dedup_input}"
    fi
done

echo "check the input file: dedup_input:" ${dedup_input}

echo "param out_file:" "${output}"
hadoop fs ${name_password} -rm -r ${output}
hadoop jar ./ime_personal.jar com.sogou.wisdsugg.feeds.FeatureMerge -D mapreduce.job.queuename=feedflow_online -D mapreduce.map.memory.mb=4096 -D mapreduce.reduce.memory.mb=4096  -D m_encoding=gbk -D output_encoding=gbk ${name_password} ${output} 100 ${dedup_input}
show_err $? ${name}
}

rand_sample_feature_list()
{
name="rand_sample_feature_list"
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'${name}"...toCheckTime"
input_dir=$1
ratio=$2
output=$3
product=$4

input="${input_dir}/${product}"
echo "check the input file: base_feature_input:" ${input}

echo "param out_file:" "${output}"
hadoop fs ${name_password} -rm -r ${output}
hadoop jar ./ime_personal.jar com.sogou.wisdsugg.feeds.FeatureRandSample -D mapreduce.job.queuename=feedflow_online -D mapreduce.map.memory.mb=256 -D mapreduce.reduce.memory.mb=256  -D m_encoding=gbk  -D output_encoding=gbk -D rand_sample_ratio=${ratio} ${name_password} ${output} 100 ${input}
show_err $? ${name}
}

merge_sample_feature()
{
name="merge_sample_feature"
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'${name}"...toCheckTime"
sample_input_dir=$1
interval=$2
output=$3
product=$4
script=$5
for (( i=0; i<=${interval}; ++i ))
do
    ts_day=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d"`
    ts_hour=`date -d"${start_hour:0:8} ${start_hour:8:10} -${i}hours" +"%Y%m%d%H"`
    sample_input="$sample_input_dir/${ts_day}/${ts_hour}/${product}"
    hadoop fs ${name_password} -test -e ${sample_input}
    if [ $? -eq 0 ];then
        base_feature_input=" -input ${sample_input} ${base_feature_input}"
    fi
done


hadoop fs ${name_password} -rm -r $3
hadoop org.apache.hadoop.streaming.HadoopStreaming \
${name_password} \
-D mapreduce.job.queuename=feedflow_online \
-D mapred.job.name="extract_feature-${model_name}-$name" \
-D mapreduce.input.fileinputformat.split.minsize=5000000000 \
-D mapreduce.map.memory.mb=4096 \
-D mapreduce.reduce.memory.mb=4096 \
${base_feature_input} -output $output -file ./script/$5 -numReduceTasks 32 -mapper "python $5 ${start_hour}"

show_err $? ${name}
}
collect_feature_list()
{
name="collect_feature_list"
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'${name}"...toCheckTime"

echo "param input:" "$1"
echo "param out_file:" "$2"
echo "param mapper_script:" "$3"
echo "param reducer_script:" $4
echo "param model_name:" $5
hadoop fs ${name_password} -rm -r $2
hadoop org.apache.hadoop.streaming.HadoopStreaming \
${name_password} \
-D mapreduce.job.queuename=feedflow_online \
-D mapred.job.name="extract_feature-${model_name}-collect_feature_list" \
-D mapreduce.input.fileinputformat.split.minsize=5000000000 \
-D mapreduce.map.memory.mb=10240 \
 $1 -output $2 -file ./script/$3 -file ./script/$4 -numReduceTasks 32 -mapper "python $3 $5" -reducer "python "$4

show_err $? ${name}
}

construct_sparse_matrix()
{
name="construct_sparse_matrix"
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'${name}"...toCheckTime"

echo "param input:" "$1"
echo "param out_file:" "$2"
echo "param mapper_script:" "$3"
echo "param feature_dir:" $4
echo "param feature_index:" $5
echo "param model_name:" $6
echo "param meta file:" $7
hadoop fs ${name_password} -rm -r $2
    hadoop org.apache.hadoop.streaming.HadoopStreaming \
    ${name_password} \
    -D mapred.job.name="extract_feature-${model_name}-construct_sparse_matrix" \
    -D mapreduce.job.queuename=feedflow_online \
    -D stream.num.map.output.key.fields=4 \
    -D num.key.fields.for.partition=4 \
    -D mapreduce.map.memory.mb=3096 \
    -D mapreduce.reduce.memory.mb=4096 \
    -D mapreduce.output.fileoutputformat.compress=true \
    -D mapreduce.output.fileoutputformat.compress.codec=com.hadoop.compression.lzo.LzopCodec \
    -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
    $1 -output $2 -file ./script/$3 -file $4/$5 -numReduceTasks 32 -mapper "python $3 $5 $6 $7"
    show_err $? ${name}
<<ok
else
    hadoop org.apache.hadoop.streaming.HadoopStreaming \
    ${name_password} \
    -D mapred.job.name="extract_feature-${model_name}-construct_sparse_matrix" \
    -D stream.num.map.output.key.fields=3 \
    -D num.key.fields.for.partition=3 \
    -D mapreduce.map.memory.mb=3096 \
    -D mapreduce.reduce.memory.mb=4096 \
    -D mapreduce.job.queuename=feedflow_online \
    -D mapreduce.output.fileoutputformat.compress=true \
    -D mapreduce.output.fileoutputformat.compress.codec=com.hadoop.compression.lzo.LzopCodec \
    -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
    $1 -output $2 -file ./script/$3 -file $4/$5 -numReduceTasks 32 -mapper "python $3 $5"
    show_err $? ${name}

fi
ok
}
construct_sparse_matrix_new()
{
name="construct_sparse_matrix"
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'${name}"...toCheckTime"

echo "param input:" "$1"
echo "param out_file:" "$2"
echo "param mapper_script:" "$3"
echo "param feature_dir:" $4
echo "param feature_index:" $5
echo "param model_name:" $6
echo "param meta file:" $7
hadoop fs ${name_password} -rm -r $2
hadoop jar ./ime_personal.jar com.sogou.wisdsugg.feeds.FeatureSparse -D mapreduce.reduce.memory.mb=256 ${name_password} ${output} 100 ${base_feature_input}

#show_err $? ${name}
}

# add by libowei  2019.09.28
construct_tfrecord()
{
name="generate_tfrecord"
echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'${name}"...toCheckTime"

echo "param input:" "$1"
echo "tfrecord output:" "$2"
echo "param reducer_out(temp):" "$3"
echo "cacheArchive:" "$4"
echo "script:" "$5"
echo "encoded_feature_list:" "$6"
echo "has_emb:" "$7"
echo "use_emb:" "$8"
echo "portrait_emb_dim:" "$9"

hadoop fs ${name_password} -rm -r $3
hadoop fs ${name_password} -rm -r $2
hadoop fs ${name_password} -mkdir -p $2

full_path=$6
feature_index=${full_path##*/}

hadoop org.apache.hadoop.streaming.HadoopStreaming \
${name_password} \
-D mapred.job.name="extract_feature-${model_name}-generate_tfrecords" \
-D mapreduce.job.queuename=feedflow_online \
-D mapreduce.map.memory.mb=4096 \
-D mapreduce.reduce.memory.mb=4096 \
-D mapreduce.input.fileinputformat.split.minsize=5000000000 \
-input $1 \
-output $3 \
-cacheArchive $4#py3 \
-file $6 \
-file ./script/$5 \
-numReduceTasks 30 \
-mapper "py3/libowei/bin/python3 $5 mapper" \
-reducer "py3/libowei/bin/python3 $5 reducer $2 ${feature_index} $7 $8 $9"

hadoop fs ${name_password} -rm -r $3

show_err $? ${name}
}
