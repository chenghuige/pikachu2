
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

# # dedup ---------------------------  not dedup
# if [ $interval != '1' ] && [ $dedup == '1' ]; then
# input=${output}
# output=${dedup_features_output}
# time dedup_features ${input} ${output} 
# hadoop fs -${name_password} -rm -r ${del_dedup_features_output}
# echo '----------------dedup_feature end' 
# fi

# # filter ---------------------------  not filter
# if [ $interval != '1' ] && [ $filter == '1' ] ; then
# input=${output}
# output=${filter_features_output}
# time filter_features ${input} ${filter_features_output}
# hadoop fs ${name_password} -rm -r ${del_filter_features_output}
# echo "----------------filter_feature end"
# fi

deal_feature_index()
{
  time sh stats-features.sh $start_hour 24
  echo "----------------stats_features end"

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

    cat ${local_featindex_dir}/${start_hour}/feature_indexes/* > ${local_featindex_dir}/${start_hour}/feature_index

    num_lines=`wc -l ${local_featindex_dir}/${start_hour}/feature_index | awk '{print $1}'`
    if (($num_lines < 3000000))
    then 
    echo "bad feature index ${local_featindex_dir}/${start_hour}/feature_index with num_lines $num_lines"
    else
      python ./scripts/merge-index-field-hash.py ${local_featindex_dir}/${start_hour}/feature_index ../../../src/conf/${TYPE}/fields.txt | iconv -f utf8 -t gbk > ${local_featindex_dir}/${start_hour}/feature_index_field &
      wait
      cp ${local_featindex_dir}/${start_hour}/feature_index_field ${local_featindex_dir}/feature_index_field.tmp
      rm -rf ${local_featindex_dir}/feature_index_field
      mv  ${local_featindex_dir}/feature_index_field.tmp ${local_featindex_dir}/feature_index_field
      rm -rf ${local_featindex_dir}/${start_hour}/feature_indexes
    fi
  }
  time copy_index
}

time deal_feature_index &

sh gen-tfrecords.sh ${start_hour} 1

sh copy-tfrecords.sh ${start_hour} 1

## in prepare-eval will use local record dir, since this is fat its ok here
## sh prepare-eval.sh  $start_hour $interval $start_hour2 $interval2 
#sh prepare-eval.sh  $start_hour 

## we can use old feature index, not much problem
# wait

echo $'\n'`date +"%Y%m%d %H:%M:%S"`$'\t'"end....deal features"



