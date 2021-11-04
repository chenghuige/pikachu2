#!/usr/bin/
## author: libowei
## note: currently should be running on 10.134.57.79

#start_hour=`date -d "-2 hours" "+%Y%m%d%H"`
start_hour="2019082612"
interval=20
model_name="sgsapp_ddu_recall_realtime_score2"

name_password="-D hadoop.client.ugi=traffic_dm,6703eb6d6dad1360d73b045f6d6b0868"

base_feature_dir="/user/traffic_dm/liuyaqiong/ctr_prediction/feature_info"
#git_branch_name=`/usr/local/git/bin/git symbolic-ref --short -q HEAD`
git_branch_name="online"
base_feature_dir="${base_feature_dir}_$git_branch_name"


sparse_matrix_output="${base_feature_dir}_${model_name}/sparse_matrix_output/${start_hour}_${interval}"
tfrecord_output="/user/traffic_dm/libowei/sparse_matrix_output_tfrecord/${start_hour}_${interval}"
temp_path="/user/traffic_dm/libowei/temp/${start_hour}_${interval}"


feature_dir="/search/odin/liuyaqiong/git_predict_online/construct_training_data/extract_feature_from_hdfs/data/feature_data_${model_name}"
#feature_list_data="${feature_dir}/${start_hour}_${interval}_feature_list"
#feature_infogain="${feature_dir}/${start_hour}_${interval}_feature_infogain"
feature_index_list="${feature_dir}/${start_hour}_${interval}_feature_index_list"

hadoop fs ${name_password} -rm -r ${temp_path}
hadoop fs ${name_password} -rm -r ${tfrecord_output}
hadoop fs ${name_password} -mkdir ${tfrecord_output}

# 阁子:
#我程序里面会在input路径下面照feature_index
#阁子:
#feature_index是预先用上面命令生成的
if [ ! -f ${feature_index_list} ]; then
  echo "${feature_index_list} does not exist!"
  exit 1
fi



cat ${feature_index_list} | iconv -f gbk -t utf8 > ./feature_index


hadoop org.apache.hadoop.streaming.HadoopStreaming \
${name_password} \
-D mapred.job.name="generate_tfrecord_test" \
-D mapreduce.job.queuename=feedflow_online \
-D mapreduce.map.memory.mb=4096 \
-D mapreduce.reduce.memory.mb=4096 \
-D mapreduce.input.fileinputformat.split.minsize=5000000000 \
-input ${sparse_matrix_output} \
-output ${temp_path} \
-cacheArchive /user/traffic_dm/libowei/python3.tgz#py3 \
-file ./feature_index \
-file ./gen-record-hdfs.py \
-numReduceTasks 32 \
-mapper "py3/libowei/bin/python3 gen-record-hdfs.py mapper" \
-reducer "py3/libowei/bin/python3 gen-record-hdfs.py reducer ${tfrecord_output} ./feature_index 0 0 0"


# 把 hdfs 上的 tfrecord 拉到本地
local_tfrecord_path="/search/odin/libowei/tfrecord"
if [ ! -d ${local_tfrecord_path} ]; then
  mkdir -p ${local_tfrecord_path}
fi
hadoop fs ${name_password} -get ${tfrecord_output} ${local_tfrecord_path}

if [ ! -d ${local_tfrecord_path}/${start_hour}_${interval} ]; then
  echo "${local_tfrecord_path}/${start_hour}_${interval} no such dir"
  exit 1
fi


# 划分训练集/验证集（随机选取一个文件作验证集)，并计数
index=0
train_num=0
valid_num=0
for file in ${local_tfrecord_path}/${start_hour}_${interval}/*
do
  #echo ${local_tfrecord_path}/${file}
  if [ $index == 0 ]
  then
    if [ ! -d "${local_tfrecord_path}/${start_hour}_${interval}/valid" ]; then
      mkdir ${local_tfrecord_path}/${start_hour}_${interval}/valid
    fi
    count=`ls ${file} | awk -F '.' '{print $3}'`
    valid_num=$(($[count]+valid_num))
    mv ${file} ${local_tfrecord_path}/${start_hour}_${interval}/valid/
  else
    if [ ! -d "${local_tfrecord_path}/${start_hour}_${interval}/train" ]; then
      mkdir ${local_tfrecord_path}/${start_hour}_${interval}/train
    fi
    count=`ls ${file} | awk -F '.' '{print $3}'`
    train_num=$(($[count]+train_num))
    mv ${file} ${local_tfrecord_path}/${start_hour}_${interval}/train/
  fi
  ((index++))
done
echo $train_num > ${local_tfrecord_path}/${start_hour}_${interval}/train/num_records.txt
echo $valid_num > ${local_tfrecord_path}/${start_hour}_${interval}/valid/num_records.txt

# 同步到 GPU 机器 (10.141.160.123)
gpu_server_ip="10.141.160.123"
remote_tf_record_path="search/odin/libowei/data/" # 去掉第一个 "/"
rsync --progress -r ${local_tfrecord_path}/${start_hour}_${interval} ${gpu_server_ip}::${remote_tf_record_path}

# 同步 feature_index 文件
rsync --progress ./feature_index ${gpu_server_ip}::${remote_tf_record_path}/${start_hour}_${interval}/

# 修改软链接
# todo

# 删除旧文件
# todo