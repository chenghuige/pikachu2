# 样本数获取

#input
model_name=$1
start_hour=$2
interval_hour=$3
train_log_hour=$4
<<ok
train_log_hour="2018092921"
interval_hour="0"
start_hour="2018092823"
model_name="nfm"
ok
is_increment=0
is_wd_sgsapp=0

if [ -z "$model_name" -o -z "$start_hour" -o -z "$interval_hour" -o  -z "$train_log_hour" ];then
    exit -1
fi

# default path
month="${train_log_hour:0:6}"
feature_base_dir="../construct_training_data/extract_feature_from_hdfs/data"
train_base_dir="../train_and_predict_model/wide_deep_model"

sample_dir="${feature_base_dir}/sparse_matrix_data_${model_name}/${start_hour}_${interval_hour}/*"
feature_info_dir="${feature_base_dir}/feature_data_${model_name}/${start_hour}_${interval_hour}_feature_infogain"
feature_index_dir="${feature_base_dir}/feature_data_${model_name}/${start_hour}_${interval_hour}_feature_index_list"
train_sample_dir="${train_base_dir}/data/train_data_${model_name}/${start_hour}_${interval_hour}_train.data"
predict_sample_dir="${train_base_dir}/data/predict_data_${model_name}/${start_hour}_${interval_hour}_predict.data"
model_dir="${train_base_dir}/data/model_data_${model_name}/model_${start_hour}_${interval_hour}.txt"
score_txt="${train_base_dir}/score/${model_name}_${month}.txt"
<<ok2
if [ ! -d "${feature_base_dir}" -o ! -d "${train_base_dir}" -o ! -d "${sample_dir}" -o ! -f "${feature_info_dir}" -o ! -f "${feature_index_dir}" -o ! -f "${predict_sample_dir}" -o ! -f "${train_sample_dir}" -o ! -f "${model_dir}" -o ! -f "${score_txt}" ];then
    exit -1
fi
ok2
#output
out_base_dir="./check_data"
check_python_shell="check_feature.py"
feature_info_out="${out_base_dir}/feature_info_index.data"
feature_model_info="${out_base_dir}/feature_info_index_model.data"
model_out="${out_base_dir}/score_get_from_model.data"
check_result="${out_base_dir}/$model_name.txt"
used_feature_category="${out_base_dir}/used_feature_category.txt"
top_20_used_feature_category="${out_base_dir}/top_20_used_feature_category.txt"
used_feature_category_ratio="${out_base_dir}/used_feature_category_ratio.txt"

# check
rm -rf ${sample_dir}lzo
total_sample_num=`wc -l ${sample_dir} | tail -1 | awk '{print $1}'`
train_sample_num=`wc -l ${train_sample_dir} | awk '{print $1}'`
predict_sample_num=`wc -l ${predict_sample_dir} | awk '{print $1}'`
train_pos_num=`awk '{print $1}' ${train_sample_dir} | sort | uniq -dci | tail -1 | awk '{print $1}'`
train_neg_num=`awk '{print $1}' ${train_sample_dir} | sort | uniq -dci | head -1 | awk '{print $1}'`
predict_pos_num=`awk '{print $1}' ${predict_sample_dir} | sort | uniq -dci | tail -1 | awk '{print $1}'`
predict_neg_num=`awk '{print $1}' ${predict_sample_dir} | sort | uniq -dci | head -1 | awk '{print $1}'`
train_ratio=`awk 'BEGIN{printf "%0.2f\n", ('$train_pos_num' / '$train_neg_num')}'`

feature_all_num=`wc -l $feature_info_dir | awk '{print $1}'`
feature_use_num=`wc -l $feature_index_dir | awk '{print $1}'`
model_predict_auc=`grep ${train_log_hour} ${score_txt} | awk '{print $5}'`
model_predict_acc=`grep ${train_log_hour} ${score_txt} | awk '{print $6}'`
 
python ${check_python_shell} ${feature_info_dir} ${feature_index_dir} ${feature_info_out}
cat ${feature_info_out} | sort -n -k1 -t $'\t'> ${feature_info_out}.sort
pre_line=`grep -n name ${model_dir} | grep "w:0" | head -1 | awk -F ":" '{print $1}'`
model_feature_num=`grep -n name ${model_dir} | grep "w:0" | head -1 | awk  '{print $3}'`
start_line=$[${pre_line} + 1]
end_line=$[${start_line} + ${model_feature_num} -1]
sed -n "${start_line},${end_line}p" ${model_dir} > ${model_out}
paste ${feature_info_out}.sort ${model_out} > ${feature_model_info}
good_feature_num=$[${model_feature_num} / 5]

awk -F '\a' '{print $1}' ${feature_model_info} | awk  '{s[$4] += $2} END{for(i in s) print i, "\t" s[i]} ' | sort > ${used_feature_category}
awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" sqrt($5*$5)}' ${feature_model_info} | sort -rn -k5  | head -${good_feature_num} | awk -F '\a' '{print $1}' | awk '{s[$4] += $2} END{for(i in s) print i,"\t" s[i]} ' | sort  > ${top_20_used_feature_category}
join -a1 ${used_feature_category} ${top_20_used_feature_category} | awk '{printf ("%s\t%.0f\t%.0f\t%.6f\n", $1, $2, $3 , $3/$2)}' | sort   > ${used_feature_category_ratio}
feature_res_name=`awk '{printf("%s\t%s\n%s\t%s\n%s\t%s\n",$1,$2,$1,$3,$1,$4)}' ${used_feature_category_ratio}|awk '{for(i=1;i<=NF;i++)a[NR,i]=$i}END{for(i=NF;i>=1;i--){for(j=1;j<=NR;j++){printf a[j,i]" "}print xxoo}}'|awk '{for(i=1;i<=NF;i++)a[NR,i]=$i}END{for(i=NF;i>=1;i--){for(j=1;j<=NR;j++){printf a[j,i]" "}print xxoo}}'|awk '{for(i=1;i<=NF;i++)a[NR,i]=$i}END{for(i=NF;i>=1;i--){for(j=1;j<=NR;j++){printf a[j,i]","}print xxoo}}'|head -1`

feature_res_value=`awk '{printf("%s\t%s\n%s\t%s\n%s\t%s\n",$1,$2,$1,$3,$1,$4)}' ${used_feature_category_ratio}|awk '{for(i=1;i<=NF;i++)a[NR,i]=$i}END{for(i=NF;i>=1;i--){for(j=1;j<=NR;j++){printf a[j,i]" "}print xxoo}}'|awk '{for(i=1;i<=NF;i++)a[NR,i]=$i}END{for(i=NF;i>=1;i--){for(j=1;j<=NR;j++){printf a[j,i]" "}print xxoo}}'|awk '{for(i=1;i<=NF;i++)a[NR,i]=$i}END{for(i=NF;i>=1;i--){for(j=1;j<=NR;j++){printf a[j,i]","}print xxoo}}'|tail -1`
echo -e "模型名称,样本结束时间,样本间隔时间,样本总数,训练样本数,训练正样本数,训练负样本数,训练正负样本比例,预测样本数,预测正样本数,预测负样本数,特征总数,使用特征数,$feature_res_name" > out
#echo -e "模型名称,样本结束时间,样本间隔时间,样本总数,训练样本数,训练正样本数,训练负样本数,训练正负样本比例,预测样本数,预测正样本数,预测负样本数,特征总数,使用特征数" > out
echo -e "$model_name,$start_hour,$interval_hour,$total_sample_num,$train_sample_num,$train_pos_num,$train_neg_num,${train_ratio},$predict_sample_num,$predict_pos_num,$predict_neg_num,${feature_all_num},${feature_use_num},$feature_res_value" >> out
#echo -e "$model_name,$start_hour,$interval_hour,$total_sample_num,$train_sample_num,$train_pos_num,$train_neg_num,${train_ratio},$predict_sample_num,$predict_pos_num,$predict_neg_num,${feature_all_num},${feature_use_num}" >> out
