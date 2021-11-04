
start_hour=$1
if [ $# == 1 ] ; then
interval=1
else
interval=$2
fi

if((`expr length $1`!=`expr length 2019100100`))
then
echo "bad input start hour "$start_hour
exit 1
fi

source ./function.sh $start_hour $interval 


input_dir=${deal_features_input}
input=$(interval_dirs $input_dir $start_hour $interval)
output=$input


hadoop fs $name_password -test -e ${encoded_feature_index_list2}
if [ $? -ne 0 ];then
    echo "[${encoded_feature_index_list2}] does not exist do nothing for gen-tfrecords.sh" 
    exit 0
fi

mkdir -p ${local_featindex_dir}/${start_hour}

copy_index()
{
  # TODO unsafe... need to lock
  # # --copy feature index list (feature.txt) to local save as feature_index
  rm -rf ${local_featindex_dir}/${start_hour}/feature_index
  rm -rf ${local_featindex_dir}/${start_hour}/feature_indexes
  mkdir -p ${local_featindex_dir}/${start_hour}/feature_indexes
  for (( i=0; i<10; ++i ))
  do
    time hadoop fs ${name_password} -copyToLocal ${encoded_feature_index_list2}/part*${i} ${local_featindex_dir}/${start_hour}/feature_indexes &
  done
  wait 
  # time hadoop fs ${name_password} -copyToLocal ${encoded_feature_index_list} ${local_featindex_dir}/feature_indexes

  cat ${local_featindex_dir}/${start_hour}/feature_indexes/* > /tmp/feature_index.${model_name}.${start_hour}

  num_lines=`wc -l /tmp/feature_index.${model_name}.${start_hour} | awk '{print $1}'`
  if (($num_lines < 3000000))
  then 
  echo "bad feature index /tmp/feature_index.{start_hour} with num_lines $num_lines"
  else
    python ./scripts/merge-index-field-hash.py /tmp/feature_index.{start_hour} | iconv -f utf8 -t gbk > /tmp/feature_index_field.${model_name}.${start_hour} 
    cp /tmp/feature_index_field.${model_name}.${start_hour}  ${local_featindex_dir}/${start_hour}/feature_index_field 
    cp ${local_featindex_dir}/${start_hour}/feature_index_field ${local_featindex_dir}/feature_index_field.tmp
    rm -rf ${local_featindex_dir}/feature_index_field
    mv  ${local_featindex_dir}/feature_index_field.tmp ${local_featindex_dir}/feature_index_field
    rm -rf ${local_featindex_dir}/${start_hour}/feature_indexes
  fi
}


time copy_index 

echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"end....deal features"



