#a=hdfs://GodSonNameNode2/user/traffic_dm/fujinbing/real_show_feature_new/20191015/2019101511/*/*
#b=/user/traffic_dm/chg/rank/tmp/utf82
input=$1
output=$2
spark-submit \
	--class com.appfeed.sogo.to_utf8 \
	--master yarn --num-executors 500 \
	--driver-memory 4g \
	--executor-cores 4 \
  --executor-memory 3g \
	--conf spark.hive.mapred.supports.subdirectories=true \
	--conf spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive=true \
	--conf spark.ui.showConsoleProgress=true \
	--conf spark.yarn.queue=root.feedflow_online \
	--conf spark.driver.maxResultSize=3g \
	--conf spark.dynamicAllocation.enabled=true \
	--conf spark.port.maxRetries=100 \
	gbk2utf8.jar \
        $input \
        $output \
        500
