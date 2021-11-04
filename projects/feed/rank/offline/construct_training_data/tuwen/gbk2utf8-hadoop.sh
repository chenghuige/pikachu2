input=$1
output=$2
hadoop org.apache.hadoop.streaming.HadoopStreaming \
            $name_password \
            -D mapred.job.name="gbk2tuf8" \
            -D mapreduce.input.fileinputformat.split.minsize=5000000000 \
            -D stream.num.map.output.key.fields=2 \
            -D num.key.fields.for.partition=2 \
            -D mapreduce.map.memory.mb=4096 \
            -D mapreduce.reduce.memory.mb=1024 \
            -D mapreduce.job.queuename=feedflow_online \
            -D mapred.reduce.tasks=0 \
            -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
            -input $input \
            -file './scripts/gbk2utf8.py' \
            -output $output \
            -mapper "python gbk2utf8.py" 

