source ./wxr_host.sh

door_time=`date +%s`
warm_interval_hour=3

# can change
file_dir_set=(
"new_rank/video"
)

host_set=(
"wxnew15.wxtop.tc.ted"
"wxnew14.wxtop.tc.ted"
"wxnew13.wxtop.tc.ted"
"wxnew12.wxtop.tc.ted"
"wxnew03.wxtop.sjs.ted"
"wxnew02.wxtop.sjs.ted"
"wxnew01.wxtop.yf.ted"
"wxnew01.wxtop.sjs.ted")

host_base_dir="odin/search/odin/daemon/wxrecserver/data"

file_set=(
"feature_index_field"
"feature_index_field.md5"
"model.pb"
"model.pb.md5"
)



err_str=""
for file_dir in ${file_dir_set[@]}
do
  cur_str=""
  for host in ${host_set[@]}
  do 
    for file in ${file_set[@]}
    do
      cur_path="${host}::${host_base_dir}/$file_dir/${file}"
      cur_path_show="${host}::$file_dir/${file}"
      cur_hour=`rsync rsync.$cur_path|awk '{print $3 " "$4}'`
      if [ ! $? -eq 0 ];then
         exit
      fi
      file_ts=`date -d "$cur_hour" +%s`
      if [  $(( $door_time - $file_ts )) -gt $(( $warm_interval_hour * 60 * 60))  ];then
        cur_str+=" $host  ${cur_hour};"
        break
      fi 
    done
  done
 if [ ! -z "$cur_str" ];then
   err_str+="$file_dir not update in such host:$cur_str"
 fi
done
if [ ! -z "$err_str" ];then
  sh -x send_xiaop_rela.sh "$err_str" 
  echo "err $err_str"
fi

#test
#time1=`rsync rsync.wxnew02.wxtop.sjs.ted::odin/search/odin/daemon/wxrecserver/data/new_rank/relative/feature_index_field.md5|awk '{print $3 " "$4}'`
#echo $time1
#date -d "$time1" +%s
